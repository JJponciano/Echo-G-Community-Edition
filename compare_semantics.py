"""Semantic comparison between two Markdown documents.

Extracts atomic facts from each document via the local LLM, embeds each fact,
and computes fuzzy match metrics (recall, precision, hallucination) using
cosine similarity over embeddings.

A "fact" is a self-contained declarative statement about a single piece of
information present in the text.

Metrics:
    recall    = matched_originals / |originals|
                  fraction of source facts represented in the reconstruction
    precision = matched_reconstructed / |reconstructed|
                  fraction of reconstructed facts that are supported by source
    hallucination_rate = 1 - precision
                  fraction of reconstructed facts that have no source support
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import ollama

from echo_g_core import DEFAULT_HOST, DEFAULT_MODEL, THINK_BLOCK

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_MATCH_THRESHOLD = 0.78
JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

FACT_SYSTEM_PROMPT = (
    "You are a semantic fact extractor. "
    "Extract every atomic factual statement from the input text that is part of "
    "the document's substantive content. "
    "Each fact must be a complete, self-contained declarative sentence in English. "
    "Resolve pronouns to their referents. "
    "Include all named entities, quantities, dates, identifiers in content, relations, and properties. "
    "EXCLUDE the following kinds of statements (they are metadata, not content): "
    "bibliography or reference entries; "
    "author lists, publication years, journal names, volume/issue/page numbers of cited works; "
    "identifiers of cited works (PMID, PMC, DOI, ISSN, ISBN, S2CID, arXiv ids, OCLC); "
    "URLs to external resources, archived links, hyperlinks; "
    "any statement of the form 'X is associated with URL Y' or 'X corresponds to URL Y'; "
    "table-of-contents entries, see-also lists, navigation labels; "
    "image captions that only repeat a label; "
    "file/path/SHA hashes. "
    "Do not invent facts. Do not omit substantive content facts. "
    "Return ONLY a single JSON array of fact strings, with no prose or fences."
)


@dataclass
class ComparisonMetrics:
    facts_original: int
    facts_reconstructed: int
    matched_originals: int
    matched_reconstructed: int
    recall: float
    precision: float
    hallucination_rate: float
    mean_match_similarity: float
    missing_facts: list[str] = field(default_factory=list)
    hallucinated_facts: list[str] = field(default_factory=list)
    matched_pairs: list[tuple[str, str, float]] = field(default_factory=list)


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    buffer: list[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) > max_chars and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = [para]
            size = len(para)
        else:
            buffer.append(para)
            size += len(para) + 2
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def parse_fact_array(raw: str) -> list[str]:
    cleaned = THINK_BLOCK.sub("", raw).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_ARRAY_RE.search(cleaned)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if isinstance(item, (str, int, float)) and str(item).strip()]


def extract_facts(text: str, *, client: ollama.Client, model: str) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for chunk in chunk_text(text):
        extra: dict[str, object] = {}
        if "qwen3" in model.lower():
            extra["think"] = False
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": FACT_SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            options={"temperature": 0.0},
            stream=False,
            **extra,
        )
        raw = response.message.content or ""
        for fact in parse_fact_array(raw):
            key = fact.lower().strip(" .")
            if key and key not in seen:
                seen.add(key)
                facts.append(fact)
    return facts


def embed_facts(facts: Sequence[str], *, client: ollama.Client, embed_model: str) -> list[list[float]]:
    if not facts:
        return []
    response = client.embed(model=embed_model, input=list(facts))
    return [list(vec) for vec in response.embeddings]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def best_match(
    query_vec: Sequence[float],
    candidate_vecs: Sequence[Sequence[float]],
) -> tuple[int, float]:
    best_idx = -1
    best_sim = -1.0
    for idx, cand in enumerate(candidate_vecs):
        sim = cosine(query_vec, cand)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx, best_sim


def compare(
    original: str,
    reconstructed: str,
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    embed_model: str = EMBED_MODEL,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> ComparisonMetrics:
    client = ollama.Client(host=host, timeout=300.0)
    logger.info("Extracting facts from original (%d chars)...", len(original))
    orig_facts = extract_facts(original, client=client, model=model)
    logger.info("Extracting facts from reconstructed (%d chars)...", len(reconstructed))
    recon_facts = extract_facts(reconstructed, client=client, model=model)
    logger.info("Embedding %d original + %d reconstructed facts...", len(orig_facts), len(recon_facts))
    orig_vecs = embed_facts(orig_facts, client=client, embed_model=embed_model)
    recon_vecs = embed_facts(recon_facts, client=client, embed_model=embed_model)

    matched_originals = 0
    matched_pairs: list[tuple[str, str, float]] = []
    missing: list[str] = []
    sims_for_matched: list[float] = []
    for fact, vec in zip(orig_facts, orig_vecs):
        if not recon_vecs:
            missing.append(fact)
            continue
        idx, sim = best_match(vec, recon_vecs)
        if sim >= threshold:
            matched_originals += 1
            matched_pairs.append((fact, recon_facts[idx], sim))
            sims_for_matched.append(sim)
        else:
            missing.append(fact)

    matched_reconstructed = 0
    hallucinated: list[str] = []
    for fact, vec in zip(recon_facts, recon_vecs):
        if not orig_vecs:
            hallucinated.append(fact)
            continue
        _, sim = best_match(vec, orig_vecs)
        if sim >= threshold:
            matched_reconstructed += 1
        else:
            hallucinated.append(fact)

    recall = matched_originals / len(orig_facts) if orig_facts else 0.0
    precision = matched_reconstructed / len(recon_facts) if recon_facts else 0.0
    halluc = 1.0 - precision if recon_facts else 0.0
    mean_sim = sum(sims_for_matched) / len(sims_for_matched) if sims_for_matched else 0.0

    return ComparisonMetrics(
        facts_original=len(orig_facts),
        facts_reconstructed=len(recon_facts),
        matched_originals=matched_originals,
        matched_reconstructed=matched_reconstructed,
        recall=recall,
        precision=precision,
        hallucination_rate=halluc,
        mean_match_similarity=mean_sim,
        missing_facts=missing,
        hallucinated_facts=hallucinated,
        matched_pairs=matched_pairs,
    )


def metrics_summary(metrics: ComparisonMetrics, threshold: float) -> dict[str, object]:
    return {
        "threshold": threshold,
        "facts_original": metrics.facts_original,
        "facts_reconstructed": metrics.facts_reconstructed,
        "matched_originals": metrics.matched_originals,
        "matched_reconstructed": metrics.matched_reconstructed,
        "recall": round(metrics.recall, 4),
        "precision": round(metrics.precision, 4),
        "hallucination_rate": round(metrics.hallucination_rate, 4),
        "mean_match_similarity": round(metrics.mean_match_similarity, 4),
        "missing_count": len(metrics.missing_facts),
        "hallucinated_count": len(metrics.hallucinated_facts),
    }


def run(
    original_path: Path,
    reconstructed_path: Path,
    *,
    output_path: Path | None,
    host: str,
    model: str,
    embed_model: str,
    threshold: float,
) -> dict[str, object]:
    original = original_path.read_text(encoding="utf-8")
    reconstructed = reconstructed_path.read_text(encoding="utf-8")
    metrics = compare(
        original,
        reconstructed,
        host=host,
        model=model,
        embed_model=embed_model,
        threshold=threshold,
    )
    summary = metrics_summary(metrics, threshold)
    report = {
        "original": str(original_path.resolve()),
        "reconstructed": str(reconstructed_path.resolve()),
        "model": model,
        "embed_model": embed_model,
        "summary": summary,
        "missing_facts": metrics.missing_facts,
        "hallucinated_facts": metrics.hallucinated_facts,
        "matched_pairs": [
            {"original": orig, "reconstructed": rec, "similarity": round(sim, 4)}
            for orig, rec, sim in metrics.matched_pairs
        ],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic comparison between two Markdown files.")
    parser.add_argument("original", type=Path)
    parser.add_argument("reconstructed", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run(
        args.original,
        args.reconstructed,
        output_path=args.output,
        host=args.host,
        model=args.model,
        embed_model=args.embed_model,
        threshold=args.threshold,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

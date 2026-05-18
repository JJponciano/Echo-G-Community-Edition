"""Standalone ECHO-G implementation.

ECHO-G means Evidence-Centered Hybrid Ontologization - Generic.
It requires a caller-provided TBox and imports no GeoMind package code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, cast

import rdflib  # doc: https://rdflib.readthedocs.io/en/stable/
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, XSD
from rdflib.term import Node

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3.6:latest"
DEFAULT_HOST = "http://localhost:11434"
ECHO = rdflib.Namespace("https://jjponciano.github.io/Echo-G/ontology/core#")
ECHOG = rdflib.Namespace("http://echo-g.local/resource#")
PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")

HASH_CHUNK_SIZE = 1 << 20
JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.I | re.S)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
LABEL_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9 _./'-]{1,80}:[ \t]*.+$")
END_SENTENCE = re.compile(r"[.!?;:]$")
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
NUMERIC_LITERAL = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
PERSON_TITLE = re.compile(
    r"^(?:dr|doctor|prof|professor|mr|mrs|ms|miss|sir|dame|nurse|clinician|"
    r"attorney|judge|engineer)\.?\s+",
    re.I,
)
IDENTIFIED_INSTANCE = re.compile(
    r"^(?:case|patient|subject|participant|asset|pump|contract|facility|loan|"
    r"invoice|claim|record|sample|device|document|project)\s+[A-Z0-9][A-Za-z0-9_.:/+-]*$",
    re.I,
)
SUPPORTED_TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}
DEFAULT_FACT_CONFIDENCE = 0.75


class EchoGError(RuntimeError):
    """Raised when ECHO-G cannot complete a conversion."""


@dataclass(frozen=True)
class SourcePageText:
    page_number: int
    text: str


PdfPageText = SourcePageText


@dataclass(frozen=True)
class Document:
    sha256: str
    text: str
    source_path: Path
    source_format: str = "unknown"


@dataclass(frozen=True)
class SourceSection:
    heading: str
    text: str
    order: int
    page_number: int


@dataclass(frozen=True)
class TBoxTerm:
    iri: str
    label: str
    kind: str
    domain: str | None = None
    range: str | None = None
    comment: str | None = None


@dataclass(frozen=True)
class TBoxSummary:
    graph: rdflib.Graph
    classes: tuple[TBoxTerm, ...]
    object_properties: tuple[TBoxTerm, ...]
    datatype_properties: tuple[TBoxTerm, ...]


@dataclass(frozen=True)
class AttributeFact:
    predicate_iri: str
    value: str
    evidence: str | None = None
    unit: str | None = None
    confidence: float | None = None
    certainty: str | None = None
    polarity: str | None = None


@dataclass(frozen=True)
class Entity:
    entity_id: str
    label: str
    class_iri: str
    evidence: str | None
    attributes: tuple[AttributeFact, ...]


@dataclass(frozen=True)
class RelationFact:
    subject_id: str
    predicate_iri: str
    object_id: str
    evidence: str | None = None
    confidence: float | None = None
    certainty: str | None = None
    polarity: str | None = None


@dataclass(frozen=True)
class OntologyProposal:
    proposal_id: str
    label: str
    kind: str
    evidence: str | None = None
    reason: str | None = None
    suggested_parent_iri: str | None = None
    suggested_domain_iri: str | None = None
    suggested_range_iri: str | None = None


@dataclass(frozen=True)
class GenericExtraction:
    document_title: str
    sections: tuple[SourceSection, ...]
    entities: tuple[Entity, ...]
    relations: tuple[RelationFact, ...]
    ontology_proposals: tuple[OntologyProposal, ...] = ()
    quality_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    heading: str
    page_number: int
    excerpt: str
    section_order: int = 1
    char_start: int = 0
    char_end: int = 0
    original_text: str | None = None


@dataclass(frozen=True)
class GraphReport:
    document_iri: str
    triple_count: int
    entity_count: int
    relation_count: int
    fact_count: int
    proposal_count: int


def ingest_source(path: Path) -> tuple[Document, tuple[SourcePageText, ...]]:
    """Hash and extract supported source documents: PDF, plain text, or Markdown."""
    if not path.exists():
        raise FileNotFoundError(f"source document not found: {path}")

    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return ingest_pdf(path)
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return ingest_text(path)
    supported = ", ".join(sorted({".pdf", *SUPPORTED_TEXT_SUFFIXES}))
    raise EchoGError(f"unsupported source type {path.suffix!r}; expected one of: {supported}")


def ingest_text(path: Path) -> tuple[Document, tuple[SourcePageText, ...]]:
    """Hash and extract a plain-text or Markdown source file."""
    sha256 = hash_file(path)
    text = path.read_text(encoding="utf-8-sig")
    source_format = "markdown" if path.suffix.casefold() in {".md", ".markdown"} else "text"
    return (
        Document(sha256=sha256, text=text, source_path=path, source_format=source_format),
        (SourcePageText(page_number=1, text=text),),
    )


def ingest_pdf(path: Path) -> tuple[Document, tuple[SourcePageText, ...]]:
    """Hash and extract a text-based PDF."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    sha256 = hash_file(path)
    pages: list[SourcePageText] = []
    import pdfplumber  # doc: https://github.com/jsvine/pdfplumber

    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            pages.append(SourcePageText(page_number=idx, text=page.extract_text() or ""))
    text = "\n".join(page.text for page in pages)
    return Document(sha256=sha256, text=text, source_path=path, source_format="pdf"), tuple(pages)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_source(source_path: Path, content_dir: Path, sha256: str) -> Path:
    content_dir.mkdir(parents=True, exist_ok=True)
    target = content_dir / sha256
    if not target.exists():
        shutil.copy2(source_path, target)
    return target


def archive_pdf(pdf_path: Path, content_dir: Path, sha256: str) -> Path:
    return archive_source(pdf_path, content_dir, sha256)


def load_tbox(path: Path, *, max_terms: int) -> TBoxSummary:
    """Load a caller-provided TBox and summarize its terms for the LLM."""
    if not path.exists():
        raise FileNotFoundError(f"TBox not found: {path}")
    graph = parse_rdf(path)
    classes = tuple(_terms(graph, OWL.Class, "class"))
    object_properties = tuple(_terms(graph, OWL.ObjectProperty, "object_property"))
    datatype_properties = tuple(_terms(graph, OWL.DatatypeProperty, "datatype_property"))

    if not classes:
        raise EchoGError(f"TBox has no owl:Class declarations: {path}")

    return TBoxSummary(
        graph=graph,
        classes=classes[:max_terms],
        object_properties=object_properties[:max_terms],
        datatype_properties=datatype_properties[:max_terms],
    )


def parse_rdf(path: Path) -> rdflib.Graph:
    graph = rdflib.Graph()
    errors: list[str] = []
    for fmt in ("turtle", "xml", "nt", "n3"):
        try:
            graph.parse(source=str(path), format=fmt)
            return graph
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
            graph = rdflib.Graph()
    raise EchoGError(f"could not parse TBox {path}: {' | '.join(errors)}")


class LocalOllama:
    """Minimal local Ollama chat client with retry on transient failures."""

    # Transient errors (connection reset, 500, timeout) are retried with
    # exponential backoff. Permanent errors (404 model not found, 400 bad
    # request) are raised on the first attempt.
    _MAX_RETRIES = 4
    _BACKOFF_BASE = 2.0  # seconds, doubled each retry

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> None:
        import ollama  # doc: https://github.com/ollama/ollama-python

        import httpx

        self.model = model
        self.host = host
        # Explicit httpx.Timeout: 30s connect, 900s read (long LLM generation)
        self._client = ollama.Client(
            host=host,
            # 240 s read timeout: fail fast on stuck generation, let retry logic
            # handle transients rather than waiting 15 min per attempt.
            timeout=httpx.Timeout(connect=30.0, read=240.0, write=30.0, pool=30.0),
        )

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        """Return True if the error looks like a transient Ollama issue."""
        import ollama  # noqa: F401

        msg = str(exc).lower()
        if isinstance(exc, ollama.ResponseError):
            # 500 = server-side crash mid-generation; 502/503/504 = upstream
            status = getattr(exc, "status_code", None)
            if status in (500, 502, 503, 504):
                return True
        # Network-level signals — connection reset, broken pipe, EOF, timeout
        return any(
            tok in msg
            for tok in (
                "wsarecv",
                "connection reset",
                "connection aborted",
                "broken pipe",
                "remotehost geschlossen",  # German Windows error message
                "remote host",
                "eof",
                "timed out",
                "timeout",
            )
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str,
        temperature: float = 0.0,
        json_format: bool = False,
    ) -> str:
        import time
        import httpx
        import ollama

        extra: dict[str, object] = {}
        if "qwen3" in self.model.lower():
            extra["think"] = False
        if json_format:
            # Grammar-constrained sampling: output is guaranteed to be valid JSON.
            extra["format"] = "json"

        last_exc: BaseException | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": temperature},
                    stream=False,
                    **extra,
                )
                return response.message.content or ""
            except (
                ollama.ResponseError,
                ConnectionError,
                OSError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                last_exc = exc
                if not self._is_transient(exc) or attempt == self._MAX_RETRIES:
                    raise
                wait = self._BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Ollama transient failure (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, self._MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        # Defensive: the loop above always either returns or raises.
        assert last_exc is not None
        raise last_exc


MAX_SINGLE_CHUNK_CHARS = 4500  # uplift switches to chunked mode beyond this


class GenericExtractor:
    """Generic local-LLM extractor constrained by a supplied TBox.

    For source documents longer than ``MAX_SINGLE_CHUNK_CHARS`` characters,
    the extractor splits the pages into chunks, extracts each chunk
    independently, then merges entity/relation/proposal sets. Entity IDs are
    prefixed per-chunk to avoid collisions, and duplicates with the same
    (label, class_iri) are coalesced.
    """

    def __init__(self, llm: LocalOllama, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._llm = llm
        self._max_attempts = max_attempts

    def extract(
        self,
        *,
        pages: tuple[SourcePageText, ...],
        tbox: TBoxSummary,
        domain_hint: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> GenericExtraction:
        if not pages:
            raise EchoGError("source document contained no extractable text")

        total_chars = sum(len(p.text) for p in pages)
        if total_chars <= MAX_SINGLE_CHUNK_CHARS:
            if progress_callback:
                progress_callback(1, 1)
            return self._extract_pages(pages, tbox=tbox, domain_hint=domain_hint)

        page_chunks = chunk_pages(pages, MAX_SINGLE_CHUNK_CHARS)
        logger.info(
            "ECHO-G chunked extraction: %d chunks (~%d chars/chunk)",
            len(page_chunks),
            total_chars // max(len(page_chunks), 1),
        )
        all_sections = parse_sections(pages)
        partials: list[tuple[int, GenericExtraction]] = []
        n_chunks = len(page_chunks)
        for idx, chunk_pages_tuple in enumerate(page_chunks, start=1):
            chunk_chars = sum(len(p.text) for p in chunk_pages_tuple)
            logger.info("ECHO-G chunk %d/%d (%d chars)", idx, n_chunks, chunk_chars)
            if progress_callback:
                progress_callback(idx, n_chunks)
            ext: GenericExtraction | None = None
            for attempt in range(1, self._max_attempts + 1):
                try:
                    ext = self._extract_pages(
                        chunk_pages_tuple, tbox=tbox, domain_hint=domain_hint
                    )
                    break
                except Exception as exc:
                    if attempt < self._max_attempts:
                        logger.warning(
                            "ECHO-G chunk %d/%d attempt %d/%d failed (%s): %s — retrying",
                            idx, n_chunks, attempt, self._max_attempts,
                            type(exc).__name__, exc,
                        )
                    else:
                        logger.warning(
                            "ECHO-G chunk %d/%d failed after %d attempts (%s): %s — skipping",
                            idx, n_chunks, self._max_attempts, type(exc).__name__, exc,
                        )
            if ext is not None:
                partials.append((idx, ext))

        if not partials:
            raise EchoGError("All extraction chunks failed")

        title = next((ext.document_title for _, ext in partials if ext.document_title), "Document")
        return merge_chunk_extractions(partials, all_sections=all_sections, title=title)

    def _extract_pages(
        self,
        pages: tuple[SourcePageText, ...],
        *,
        tbox: TBoxSummary,
        domain_hint: str,
    ) -> GenericExtraction:
        sections = parse_sections(pages)
        source_text = "\n".join(page.text for page in pages)
        failures: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            logger.info("ECHO-G extraction attempt %d/%d", attempt, self._max_attempts)
            raw = self._llm.complete(
                build_prompt(
                    source_text=source_text,
                    sections=sections,
                    tbox=tbox,
                    domain_hint=domain_hint,
                    failures=tuple(failures),
                ),
                system=system_prompt(),
                json_format=True,
            )
            try:
                extraction = payload_to_extraction(parse_json_object(raw), sections, tbox)
            except EchoGError as exc:
                failures.append(str(exc))
                continue

            issues = quality_issues(extraction, sections)
            if not issues:
                return extraction
            failures.append("; ".join(issues))

        raise EchoGError(
            f"LLM extraction failed after {self._max_attempts} attempt(s): {' | '.join(failures)}"
        )


def split_page_text(page: SourcePageText, max_chars: int) -> list[SourcePageText]:
    """Split a single page into sub-pages on paragraph/line boundaries."""
    if len(page.text) <= max_chars:
        return [page]
    parts: list[str] = []
    paragraphs = page.text.split("\n\n")
    buffer: list[str] = []
    size = 0
    for para in paragraphs:
        if buffer and size + len(para) + 2 > max_chars:
            parts.append("\n\n".join(buffer))
            buffer = [para]
            size = len(para)
        else:
            buffer.append(para)
            size += len(para) + 2
    if buffer:
        parts.append("\n\n".join(buffer))
    # If any single paragraph still exceeds the limit, split it by lines.
    final: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
            continue
        lines = part.splitlines()
        buf: list[str] = []
        sz = 0
        for line in lines:
            if buf and sz + len(line) + 1 > max_chars:
                final.append("\n".join(buf))
                buf = [line]
                sz = len(line)
            else:
                buf.append(line)
                sz += len(line) + 1
        if buf:
            final.append("\n".join(buf))
    return [SourcePageText(page_number=page.page_number, text=chunk) for chunk in final if chunk.strip()]


def chunk_pages(
    pages: tuple[SourcePageText, ...], max_chars: int
) -> list[tuple[SourcePageText, ...]]:
    """Group pages into chunks of at most ``max_chars`` characters.

    Pages that individually exceed ``max_chars`` are first sub-split on
    paragraph and line boundaries.
    """
    expanded: list[SourcePageText] = []
    for page in pages:
        expanded.extend(split_page_text(page, max_chars))

    chunks: list[tuple[SourcePageText, ...]] = []
    buffer: list[SourcePageText] = []
    size = 0
    for page in expanded:
        page_len = len(page.text)
        if buffer and size + page_len > max_chars:
            chunks.append(tuple(buffer))
            buffer = [page]
            size = page_len
        else:
            buffer.append(page)
            size += page_len
    if buffer:
        chunks.append(tuple(buffer))
    return chunks


def merge_chunk_extractions(
    partials: list[tuple[int, GenericExtraction]],
    *,
    all_sections: tuple[SourceSection, ...],
    title: str,
) -> GenericExtraction:
    """Merge per-chunk extractions, deduping entities by (label, class_iri)."""
    entities: list[Entity] = []
    relations: list[RelationFact] = []
    proposals: list[OntologyProposal] = []
    quality: list[str] = []
    seen_entities: dict[tuple[str, str], str] = {}
    seen_relations: set[tuple[str, str, str]] = set()
    seen_proposals: set[tuple[str, str]] = set()

    for chunk_idx, ext in partials:
        local_to_canonical: dict[str, str] = {}
        for entity in ext.entities:
            key = (entity.label.lower().strip(), entity.class_iri)
            if key in seen_entities:
                local_to_canonical[entity.entity_id] = seen_entities[key]
                continue
            canonical_id = f"c{chunk_idx}_{entity.entity_id}"
            seen_entities[key] = canonical_id
            local_to_canonical[entity.entity_id] = canonical_id
            entities.append(
                Entity(
                    entity_id=canonical_id,
                    label=entity.label,
                    class_iri=entity.class_iri,
                    evidence=entity.evidence,
                    attributes=entity.attributes,
                )
            )
        for relation in ext.relations:
            subj = local_to_canonical.get(relation.subject_id, f"c{chunk_idx}_{relation.subject_id}")
            obj = local_to_canonical.get(relation.object_id, f"c{chunk_idx}_{relation.object_id}")
            rkey = (subj, relation.predicate_iri, obj)
            if rkey in seen_relations:
                continue
            seen_relations.add(rkey)
            relations.append(
                RelationFact(
                    subject_id=subj,
                    predicate_iri=relation.predicate_iri,
                    object_id=obj,
                    evidence=relation.evidence,
                    confidence=relation.confidence,
                    certainty=relation.certainty,
                    polarity=relation.polarity,
                )
            )
        for proposal in ext.ontology_proposals:
            pkey = (proposal.label.lower().strip(), proposal.kind)
            if pkey in seen_proposals:
                continue
            seen_proposals.add(pkey)
            proposals.append(
                OntologyProposal(
                    proposal_id=f"c{chunk_idx}_{proposal.proposal_id}",
                    label=proposal.label,
                    kind=proposal.kind,
                    evidence=proposal.evidence,
                    reason=proposal.reason,
                    suggested_parent_iri=proposal.suggested_parent_iri,
                    suggested_domain_iri=proposal.suggested_domain_iri,
                    suggested_range_iri=proposal.suggested_range_iri,
                )
            )
        quality.extend(ext.quality_issues)

    return GenericExtraction(
        document_title=title,
        sections=all_sections,
        entities=tuple(entities),
        relations=tuple(relations),
        ontology_proposals=tuple(proposals),
        quality_issues=tuple(quality),
    )


def parse_sections(pages: tuple[SourcePageText, ...]) -> tuple[SourceSection, ...]:
    """Split source text into generic sections using Markdown and heading heuristics."""
    sections: list[SourceSection] = []
    current_heading = "Document"
    current_page = pages[0].page_number if pages else 1
    current_lines: list[str] = []
    title_seen = False

    def flush() -> None:
        nonlocal current_lines
        text = clean_multiline("\n".join(current_lines))
        if text or current_heading:
            sections.append(SourceSection(current_heading, text, len(sections) + 1, current_page))
        current_lines = []

    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                current_lines.append(raw_line)
                continue
            markdown_title = markdown_heading(line)
            if markdown_title:
                if not title_seen:
                    title_seen = True
                    current_heading = markdown_title[:80]
                    current_page = page.page_number
                    continue
                flush()
                current_heading = markdown_title[:80]
                current_page = page.page_number
                continue
            if not title_seen:
                title_seen = True
                current_heading = line[:80]
                current_page = page.page_number
                continue
            if looks_like_heading(line):
                flush()
                current_heading = line
                current_page = page.page_number
                continue
            current_lines.append(raw_line)
    flush()

    if not sections:
        text = "\n".join(page.text for page in pages).strip()
        return (SourceSection("Document", text, 1, pages[0].page_number if pages else 1),)
    return tuple(sections)


def markdown_heading(line: str) -> str | None:
    match = MARKDOWN_HEADING.match(line)
    if match is None:
        return None
    heading = compact(match.group(2).strip())
    return heading or None


def looks_like_heading(line: str) -> bool:
    if markdown_heading(line):
        return True
    if len(line) > 80 or len(line.split()) > 8:
        return False
    if not line[0].isalpha() or LABEL_LINE.match(line):
        return False
    if END_SENTENCE.search(line):
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return uppercase_ratio > 0.18 or line.istitle()


def system_prompt() -> str:
    return (
        "You are ECHO-G, a local generic ontology extraction engine. "
        "Use only the supplied TBox IRIs. Never invent class or property IRIs. "
        "Extract only facts explicitly present in the source text. Build a connected graph, "
        "not a flat list of section summaries. "
        "Return exactly one JSON object and no prose."
    )


def build_prompt(
    *,
    source_text: str,
    sections: tuple[SourceSection, ...],
    tbox: TBoxSummary,
    domain_hint: str,
    failures: tuple[str, ...],
) -> str:
    schema = {
        "document_title": "string",
        "entities": [
            {
                "id": "short stable id",
                "label": "source label",
                "class_iri": "must be one supplied TBox class IRI",
                "evidence": "short source excerpt",
                "attributes": [
                    {
                        "predicate_iri": "must be supplied TBox datatype property IRI",
                        "value": "literal value",
                        "unit": "optional unit text exactly present near the value",
                        "confidence": "number from 0 to 1",
                        "certainty": "certain|probable|possible|unverified",
                        "polarity": "affirmed|negated",
                        "evidence": "short source excerpt",
                    }
                ],
            }
        ],
        "relations": [
            {
                "subject_id": "entity id",
                "predicate_iri": "must be supplied TBox object property IRI",
                "object_id": "entity id",
                "confidence": "number from 0 to 1",
                "certainty": "certain|probable|possible|unverified",
                "polarity": "affirmed|negated",
                "evidence": "short source excerpt",
            }
        ],
        "ontology_proposals": [
            {
                "id": "short stable id",
                "label": "missing concept/property label",
                "kind": "class|object_property|datatype_property",
                "evidence": "short source excerpt",
                "reason": "why supplied TBox does not cover it",
                "suggested_parent_iri": "optional supplied parent class/property IRI",
                "suggested_domain_iri": "optional supplied domain class IRI",
                "suggested_range_iri": "optional supplied range class/datatype IRI",
            }
        ],
    }
    blocks = [
        f"Domain/task hint: {domain_hint or 'generic document'}",
        "",
        "Extraction rules:",
        "- Represent specific people, organizations, cases, observations, measurements, "
        "claims, decisions, actions, plans, outcomes, instruments, and records as separate "
        "entities when they are explicit in the text.",
        "- Connect entities with supplied object properties. Prefer generic supplied "
        "properties such as concerns, about, hasParticipant, performedBy, hasResult, "
        "supportsClaim, hasOutcome, hasRisk, assignedTo, decidedBy, usesMethod, and "
        "usesInstrument when a domain-specific property is not available.",
        "- Use supplied datatype properties for literal attributes only. Do not create "
        "properties such as hasDiagnosis or hasTreatmentPlan unless they are listed below.",
        "- If a source concept, attribute, or relation is explicit but no supplied TBox term "
        "maps to it well, do not invent an IRI. Add it to ontology_proposals as a proposed "
        "class, object_property, or datatype_property addition.",
        "- TBox class proposals must be reusable types, not named individuals. A person, "
        "organization, patient, case, device, asset, contract, sample, or record with a "
        "specific name/code belongs in entities as an ABox instance. For example, "
        "'Dr. Jane Smith' is an entity typed with the closest supplied class such as Person, "
        "Agent, Clinician, or Entity; it is never a class proposal. Only propose reusable "
        "classes such as Clinician, Physician, Pump, Clause, Facility, or Biomarker.",
        "- If a named instance needs a more specific missing class, keep the named instance "
        "as an entity using the closest generic supplied class and propose only the missing "
        "generic type, not the individual's name.",
        "- If you map a specific source concept to a very generic supplied term because the "
        "domain term is missing, also add an ontology_proposal for the missing specific term.",
        "- Evidence must be a short exact excerpt from the source text.",
        "- Avoid using an entire section as one entity when the section contains multiple "
        "facts; split it into the smallest meaningful entities and relations.",
        "",
        "JSON schema:",
        json.dumps(schema, indent=2),
        "",
        "Supplied TBox terms:",
        tbox_prompt_block(tbox),
        "",
        "Parser source sections:",
        json.dumps([asdict(section) for section in sections], indent=2, ensure_ascii=False),
        "",
        "Source text:",
        source_text,
    ]
    if failures:
        blocks.extend(["", "Previous attempt was rejected:", "\n".join(f"- {x}" for x in failures)])
    return "\n".join(blocks)


def tbox_prompt_block(tbox: TBoxSummary) -> str:
    parts = ["Classes:"]
    parts.extend(_term_line(term) for term in tbox.classes)
    parts.append("")
    parts.append("Object properties:")
    parts.extend(_term_line(term) for term in tbox.object_properties)
    parts.append("")
    parts.append("Datatype properties:")
    parts.extend(_term_line(term) for term in tbox.datatype_properties)
    return "\n".join(parts)


def parse_json_object(raw: str) -> Mapping[str, object]:
    stripped = THINK_BLOCK.sub("", raw).strip()
    fence = JSON_FENCE.match(stripped)
    if fence is not None:
        stripped = fence.group(1).strip()
    try:
        parsed: object = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise EchoGError("LLM output did not contain a JSON object") from None
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise EchoGError(f"expected JSON object, got {type(parsed).__name__}")
    return cast(Mapping[str, object], parsed)


def payload_to_extraction(
    payload: Mapping[str, object],
    sections: tuple[SourceSection, ...],
    tbox: TBoxSummary,
) -> GenericExtraction:
    object_property_iris = {term.iri for term in tbox.object_properties}
    datatype_property_iris = {term.iri for term in tbox.datatype_properties}
    entities: list[Entity] = []
    seen_ids: set[str] = set()
    rejections: list[str] = []
    proposals: list[OntologyProposal] = []

    for item in list_of_mappings(payload.get("ontology_proposals")):
        try:
            proposals.append(ontology_proposal_from_mapping(item))
        except EchoGError as exc:
            rejections.append(str(exc))

    for item in list_of_mappings(payload.get("entities")):
        try:
            entity, entity_proposals = entity_from_mapping(
                item,
                tbox,
                datatype_property_iris,
            )
        except EchoGError as exc:
            rejections.append(str(exc))
            proposals.extend(proposals_from_rejected_entity(item, str(exc)))
            continue
        proposals.extend(entity_proposals)
        if entity.entity_id in seen_ids:
            rejections.append(f"duplicate entity id: {entity.entity_id}")
            continue
        seen_ids.add(entity.entity_id)
        entities.append(entity)

    relations: list[RelationFact] = []
    for item in list_of_mappings(payload.get("relations")):
        try:
            relation = relation_from_mapping(item, seen_ids, object_property_iris)
        except EchoGError as exc:
            rejections.append(str(exc))
            proposals.extend(proposals_from_rejected_relation(item, str(exc)))
            continue
        relations.append(relation)

    title = string_or_none(payload.get("document_title")) or sections[0].heading
    deduped_proposals = deduplicate_proposals(proposals)
    return GenericExtraction(
        document_title=title,
        sections=sections,
        entities=tuple(entities),
        relations=tuple(relations),
        ontology_proposals=deduped_proposals,
        quality_issues=tuple(rejections),
    )


def entity_from_mapping(
    item: Mapping[str, object],
    tbox: TBoxSummary,
    datatype_property_iris: set[str],
) -> tuple[Entity, tuple[OntologyProposal, ...]]:
    class_iris = {term.iri for term in tbox.classes}
    entity_id = slug(string_or_none(item.get("id")) or string_or_none(item.get("label")) or "")
    label = string_or_none(item.get("label")) or entity_id
    class_iri = iri_or_none(item.get("class_iri")) or ""
    proposals: list[OntologyProposal] = []
    if class_iri not in class_iris:
        fallback_class = fallback_class_iri_for_entity(label, class_iri, tbox)
        if fallback_class is None:
            raise EchoGError(f"unknown class_iri for entity {entity_id}: {class_iri}")
        proposals.extend(
            proposals_from_unknown_entity_class(
                item,
                class_iri,
                reason=(
                    f"Entity used unknown class IRI {class_iri}; kept the source entity as "
                    f"an instance of {fallback_class}."
                ),
                suggested_parent_iri=fallback_class,
            )
        )
        class_iri = fallback_class
    attributes: list[AttributeFact] = []
    for attribute in list_of_mappings(item.get("attributes")):
        try:
            attributes.append(attribute_from_mapping(attribute, datatype_property_iris))
        except EchoGError as exc:
            logger.debug("Rejected invalid attribute for entity %s: %r", entity_id, attribute)
            proposals.extend(proposals_from_rejected_attribute(attribute, entity_id, str(exc)))
    return (
        Entity(
            entity_id=entity_id,
            label=label,
            class_iri=class_iri,
            evidence=string_or_none(item.get("evidence")),
            attributes=tuple(attributes),
        ),
        tuple(proposals),
    )


def attribute_from_mapping(
    item: Mapping[str, object],
    datatype_property_iris: set[str],
) -> AttributeFact:
    predicate = iri_or_none(item.get("predicate_iri")) or ""
    value = string_or_none(item.get("value")) or ""
    if predicate not in datatype_property_iris:
        raise EchoGError(f"unknown datatype predicate_iri: {predicate}")
    if not value:
        raise EchoGError(f"empty literal value for datatype predicate: {predicate}")
    return AttributeFact(
        predicate_iri=predicate,
        value=value,
        evidence=string_or_none(item.get("evidence")),
        unit=string_or_none(item.get("unit")),
        confidence=float_or_none(item.get("confidence")),
        certainty=string_or_none(item.get("certainty")),
        polarity=string_or_none(item.get("polarity")),
    )


def relation_from_mapping(
    item: Mapping[str, object],
    entity_ids: set[str],
    object_property_iris: set[str],
) -> RelationFact:
    subject_id = slug(string_or_none(item.get("subject_id")) or "")
    object_id = slug(string_or_none(item.get("object_id")) or "")
    predicate = iri_or_none(item.get("predicate_iri")) or ""
    if subject_id not in entity_ids:
        raise EchoGError(f"relation subject_id is unknown: {subject_id}")
    if object_id not in entity_ids:
        raise EchoGError(f"relation object_id is unknown: {object_id}")
    if predicate not in object_property_iris:
        raise EchoGError(f"unknown object predicate_iri: {predicate}")
    return RelationFact(
        subject_id=subject_id,
        predicate_iri=predicate,
        object_id=object_id,
        evidence=string_or_none(item.get("evidence")),
        confidence=float_or_none(item.get("confidence")),
        certainty=string_or_none(item.get("certainty")),
        polarity=string_or_none(item.get("polarity")),
    )


def ontology_proposal_from_mapping(item: Mapping[str, object]) -> OntologyProposal:
    label = (
        string_or_none(item.get("label"))
        or string_or_none(item.get("name"))
        or string_or_none(item.get("concept"))
    )
    kind = normalized_proposal_kind(string_or_none(item.get("kind")))
    if label is None:
        raise EchoGError("ontology proposal is missing label")
    if kind is None:
        raise EchoGError(f"ontology proposal has invalid kind for {label!r}")
    if kind == "class" and looks_like_instance_label(label):
        raise EchoGError(f"ontology proposal is an instance, not a reusable class: {label}")
    proposal_id = slug(string_or_none(item.get("id")) or f"{kind}_{label}")
    return OntologyProposal(
        proposal_id=proposal_id,
        label=label,
        kind=kind,
        evidence=string_or_none(item.get("evidence")),
        reason=string_or_none(item.get("reason")),
        suggested_parent_iri=iri_or_none(item.get("suggested_parent_iri")),
        suggested_domain_iri=iri_or_none(item.get("suggested_domain_iri")),
        suggested_range_iri=iri_or_none(item.get("suggested_range_iri")),
    )


def fallback_class_iri_for_entity(
    label: str,
    requested_class_iri: str,
    tbox: TBoxSummary,
) -> str | None:
    target = f"{label} {humanize_label(local_name(requested_class_iri))}".casefold()
    preferred_groups: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if PERSON_TITLE.search(target):
        preferred_groups.append(
            (
                ("clinician", "physician", "doctor", "nurse", "provider", "person", "agent"),
                ("party", "engineer", "human reviewer"),
            )
        )
    if re.search(r"\bpatient\b", target):
        preferred_groups.append((("patient", "person", "agent"), ("entity", "resource")))
    if re.search(r"\bcase\b", target):
        preferred_groups.append((("case",), ("record", "resource", "entity")))
    if re.search(r"\b(contract|agreement)\b", target):
        preferred_groups.append((("contract",), ("record", "resource", "entity")))
    if re.search(r"\b(pump|asset|device|instrument|equipment|facility)\b", target):
        preferred_groups.append(
            (("asset", "device", "instrument", "facility"), ("entity", "resource"))
        )
    if re.search(r"\b(measurement|temperature|pressure|reading|value|dose|weight|ratio)\b", target):
        preferred_groups.append((("measurement", "observation", "value"), ("entity", "resource")))
    preferred_groups.append((("entity", "resource"), ()))

    for primary, secondary in preferred_groups:
        found = first_class_matching(tbox.classes, primary)
        if found:
            return found
        found = first_class_matching(tbox.classes, secondary)
        if found:
            return found
    return tbox.classes[0].iri if tbox.classes else None


def first_class_matching(terms: tuple[TBoxTerm, ...], labels: tuple[str, ...]) -> str | None:
    wanted = {label.casefold() for label in labels if label}
    if not wanted:
        return None
    for term in terms:
        names = {
            compact(term.label).casefold(),
            compact(humanize_label(local_name(term.iri))).casefold(),
        }
        if names & wanted:
            return term.iri
    for term in terms:
        haystack = f"{term.label} {humanize_label(local_name(term.iri))}".casefold()
        if any(label in haystack for label in wanted):
            return term.iri
    return None


def proposals_from_unknown_entity_class(
    item: Mapping[str, object],
    class_iri: str,
    *,
    reason: str,
    suggested_parent_iri: str | None = None,
) -> tuple[OntologyProposal, ...]:
    label = reusable_class_label_from_entity(item, class_iri)
    if label is None:
        return ()
    return (
        OntologyProposal(
            proposal_id=slug(f"class_{label}"),
            label=label,
            kind="class",
            evidence=string_or_none(item.get("evidence")),
            reason=reason,
            suggested_parent_iri=suggested_parent_iri,
        ),
    )


def reusable_class_label_from_entity(
    item: Mapping[str, object],
    class_iri: str,
) -> str | None:
    class_label = humanize_label(local_name(class_iri))
    entity_label = string_or_none(item.get("label"))
    for candidate in (class_label, entity_label):
        if candidate and not looks_like_instance_label(candidate):
            return candidate
    return None


def proposals_from_rejected_entity(
    item: Mapping[str, object],
    reason: str,
) -> tuple[OntologyProposal, ...]:
    class_iri = iri_or_none(item.get("class_iri"))
    if not class_iri or "unknown class_iri" not in reason:
        return ()
    return proposals_from_unknown_entity_class(
        item,
        class_iri,
        reason=f"Entity used unknown class IRI {class_iri}; {reason}",
    )


def proposals_from_rejected_attribute(
    item: Mapping[str, object],
    entity_id: str,
    reason: str,
) -> tuple[OntologyProposal, ...]:
    predicate = iri_or_none(item.get("predicate_iri"))
    if not predicate or "unknown datatype predicate_iri" not in reason:
        return ()
    label = local_name(predicate)
    return (
        OntologyProposal(
            proposal_id=slug(f"datatype_property_{label}_{entity_id}"),
            label=label,
            kind="datatype_property",
            evidence=string_or_none(item.get("evidence")),
            reason=f"Attribute for entity {entity_id} used unknown datatype property; {reason}",
        ),
    )


def proposals_from_rejected_relation(
    item: Mapping[str, object],
    reason: str,
) -> tuple[OntologyProposal, ...]:
    predicate = iri_or_none(item.get("predicate_iri"))
    if not predicate or "unknown object predicate_iri" not in reason:
        return ()
    label = local_name(predicate)
    return (
        OntologyProposal(
            proposal_id=slug(f"object_property_{label}"),
            label=label,
            kind="object_property",
            evidence=string_or_none(item.get("evidence")),
            reason=f"Relation used unknown object property; {reason}",
        ),
    )


def deduplicate_proposals(
    proposals: Iterable[OntologyProposal],
) -> tuple[OntologyProposal, ...]:
    by_key: dict[tuple[str, str, str | None, str | None, str | None], OntologyProposal] = {}
    for proposal in proposals:
        key = (
            proposal.kind,
            proposal.label.casefold(),
            proposal.suggested_parent_iri,
            proposal.suggested_domain_iri,
            proposal.suggested_range_iri,
        )
        if key not in by_key:
            by_key[key] = proposal
    return tuple(sorted(by_key.values(), key=lambda item: (item.kind, item.label.casefold())))


def normalized_proposal_kind(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "class": "class",
        "concept": "class",
        "entity_class": "class",
        "object_property": "object_property",
        "relation": "object_property",
        "relationship": "object_property",
        "datatype_property": "datatype_property",
        "data_property": "datatype_property",
        "attribute": "datatype_property",
        "literal_property": "datatype_property",
    }
    return aliases.get(key)


def quality_issues(
    extraction: GenericExtraction,
    sections: tuple[SourceSection, ...],
) -> tuple[str, ...]:
    issues: list[str] = []
    if not extraction.document_title:
        issues.append("missing document_title")
    if len(extraction.sections) != len(sections):
        issues.append("source sections were not preserved")
    if len(extraction.entities) > 1 and not extraction.relations:
        issues.append(
            "extraction is flat: connect related entities with supplied object properties"
        )
    over_aggregated = [
        entity.entity_id
        for entity in extraction.entities
        if len(entity.label.split()) > 18 and len(entity.attributes) > 1
    ]
    if over_aggregated:
        issues.append(
            "over-aggregated entities should be split into smaller entities: "
            + ", ".join(over_aggregated[:5])
        )
    return tuple(issues)


def build_ontology(
    *,
    document: Document,
    extraction: GenericExtraction,
    tbox: TBoxSummary,
    model_name: str | None = None,
    extracted_at: datetime | None = None,
) -> tuple[rdflib.Graph, GraphReport]:
    graph = rdflib.Graph()
    bind_namespaces(graph)
    for prefix, namespace in tbox.graph.namespaces():
        graph.bind(prefix, namespace, replace=False)
    add_core_vocabulary(graph)
    for triple in tbox.graph:
        graph.add(triple)

    doc = ECHOG[f"document_{document.sha256[:16]}"]
    graph.add((doc, RDF.type, ECHOG.ProcessedDocument))
    graph.add((doc, RDF.type, ECHO.Document))
    graph.add((doc, RDFS.label, rdflib.Literal(extraction.document_title)))
    graph.add((doc, DCTERMS.title, rdflib.Literal(extraction.document_title)))
    graph.add((doc, DCTERMS.format, rdflib.Literal(document.source_format)))
    graph.add((doc, ECHO.hasTitle, rdflib.Literal(extraction.document_title)))
    graph.add((doc, ECHO.hasSha256, rdflib.Literal(document.sha256)))
    graph.add((doc, ECHO.hasSourceFile, rdflib.Literal(str(document.source_path.resolve()))))
    graph.add((doc, ECHO.hasOriginalText, rdflib.Literal(document.text)))
    graph.add((doc, DCTERMS.source, rdflib.URIRef(document.source_path.resolve().as_uri())))
    run = add_extraction_run(
        graph,
        document,
        model_name=model_name,
        extracted_at=extracted_at,
    )

    entity_nodes: dict[str, rdflib.URIRef] = {}
    section_nodes: dict[int, rdflib.URIRef] = {}
    fact_count = 0
    for section in extraction.sections:
        section_node = ECHOG[f"section_{document.sha256[:8]}_{section.order}"]
        section_nodes[section.order] = section_node
        graph.add((section_node, RDF.type, ECHO.DocumentSection))
        graph.add((doc, ECHO.hasSection, section_node))
        graph.add((section_node, RDFS.label, rdflib.Literal(section.heading)))
        graph.add((section_node, ECHO.hasHeading, rdflib.Literal(section.heading)))
        graph.add((section_node, ECHO.hasTitle, rdflib.Literal(section.heading)))
        graph.add((section_node, ECHO.hasText, rdflib.Literal(section.text)))
        graph.add((section_node, ECHO.hasOriginalText, rdflib.Literal(section.text)))
        graph.add(
            (section_node, ECHO.hasOrder, rdflib.Literal(section.order, datatype=XSD.integer))
        )
        graph.add(
            (
                section_node,
                ECHO.hasPageNumber,
                rdflib.Literal(section.page_number, datatype=XSD.integer),
            )
        )

    for entity in extraction.entities:
        node = ECHOG[f"entity_{document.sha256[:8]}_{entity.entity_id}"]
        entity_nodes[entity.entity_id] = node
        add_assertion(
            graph,
            node,
            RDF.type,
            rdflib.URIRef(entity.class_iri),
            extraction,
            document_node=doc,
            section_nodes=section_nodes,
            run_node=run,
            evidence_text=entity.evidence,
        )
        add_assertion(
            graph,
            node,
            RDFS.label,
            rdflib.Literal(entity.label),
            extraction,
            document_node=doc,
            section_nodes=section_nodes,
            run_node=run,
            evidence_text=entity.evidence,
        )
        graph.add((doc, ECHO.hasEntity, node))
        graph.add((doc, ECHO.mentions, node))
        for attribute in entity.attributes:
            add_assertion(
                graph,
                node,
                rdflib.URIRef(attribute.predicate_iri),
                rdflib.Literal(attribute.value),
                extraction,
                document_node=doc,
                section_nodes=section_nodes,
                run_node=run,
                evidence_text=attribute.evidence,
                confidence=attribute.confidence,
                certainty=attribute.certainty,
                polarity=attribute.polarity,
                unit_text=attribute.unit,
            )
            fact_count += 1

    for relation in extraction.relations:
        add_assertion(
            graph,
            entity_nodes[relation.subject_id],
            rdflib.URIRef(relation.predicate_iri),
            entity_nodes[relation.object_id],
            extraction,
            document_node=doc,
            section_nodes=section_nodes,
            run_node=run,
            evidence_text=relation.evidence,
            confidence=relation.confidence,
            certainty=relation.certainty,
            polarity=relation.polarity,
        )
        fact_count += 1

    for proposal in extraction.ontology_proposals:
        add_ontology_proposal(
            graph,
            proposal,
            extraction,
            document_node=doc,
            section_nodes=section_nodes,
            run_node=run,
        )

    return graph, GraphReport(
        document_iri=str(doc),
        triple_count=len(graph),
        entity_count=len(extraction.entities),
        relation_count=len(extraction.relations),
        fact_count=fact_count,
        proposal_count=len(extraction.ontology_proposals),
    )


def reconstruct_markdown(
    graph: rdflib.Graph,
    document_iri: str | None = None,
    *,
    include_metadata: bool = False,
    include_entities: bool = False,
    include_relations: bool = False,
    include_proposals: bool = False,
) -> str:
    """Reconstruct a narrative Markdown view of the document.

    By default, only the source sections and a compact relations list are
    emitted. The other channels (document metadata, structured entity dump,
    ontology proposals) are dev-facing and would pollute a semantic-round-trip
    comparison against the original.
    """
    doc = select_document(graph, document_iri)
    title = literal_any(graph, doc, (RDFS.label, ECHO.hasTitle, DCTERMS.title)) or "ECHO-G Document"
    lines = [f"# {title}", ""]

    if include_metadata:
        source = literal_any(graph, doc, (ECHO.hasSourceFile, ECHOG.sourcePath))
        sha256 = literal_any(graph, doc, (ECHO.hasSha256, ECHOG.sourceSha256))
        if source or sha256:
            lines.append("## Document Metadata")
            if source:
                lines.append(f"- Source: {source}")
            if sha256:
                lines.append(f"- SHA-256: {sha256}")
            lines.append("")

    lines.extend(source_sections_markdown(graph, doc))
    if include_entities:
        lines.extend(entities_markdown(graph, doc))
    if include_relations:
        lines.extend(relations_markdown(graph))
    if include_proposals:
        lines.extend(ontology_proposals_markdown(graph, doc))
    return "\n".join(lines).rstrip() + "\n"


def source_sections_markdown(graph: rdflib.Graph, doc: rdflib.URIRef) -> list[str]:
    sections = sorted(
        uris_any(graph, doc, (ECHO.hasSection, ECHOG.hasSection)),
        key=lambda node: int(literal_any(graph, node, (ECHO.hasOrder, ECHOG.sectionOrder)) or "0"),
    )
    if not sections:
        return []
    lines = ["## Source Sections"]
    for section in sections:
        heading = literal_any(graph, section, (ECHO.hasHeading, ECHOG.sectionHeading)) or "Section"
        text = literal_any(graph, section, (ECHO.hasText, ECHOG.sectionText)) or ""
        page = literal_any(graph, section, (ECHO.hasPageNumber, ECHOG.pageNumber))
        suffix = f" (page {page})" if page else ""
        lines.extend(["", f"### {heading}{suffix}", "", text])
    lines.append("")
    return lines


def entities_markdown(graph: rdflib.Graph, doc: rdflib.URIRef) -> list[str]:
    entities = uris_any(graph, doc, (ECHO.hasEntity, ECHOG.hasEntity))
    if not entities:
        return []
    lines = ["## Extracted Entities", ""]
    for entity in sorted(entities, key=lambda node: literal(graph, node, RDFS.label) or str(node)):
        label = literal(graph, entity, RDFS.label) or local_name(str(entity))
        class_iri = first_non_core_type(graph, entity)
        class_label = label_for_iri(graph, class_iri) if class_iri else None
        lines.append(f"### {label}")
        lines.append(f"- Class: {class_label or class_iri or 'unknown'}")
        for predicate, value in literal_attributes(graph, entity):
            predicate_label = label_for_iri(graph, predicate) or predicate
            lines.append(f"- {predicate_label}: {value}")
        lines.append("")
    return lines


def relations_markdown(graph: rdflib.Graph) -> list[str]:
    rows: list[str] = []
    for fact in extracted_fact_nodes(graph):
        subject = first_uri(graph, cast(rdflib.URIRef, fact), RDF.subject)
        predicate = first_uri(graph, cast(rdflib.URIRef, fact), RDF.predicate)
        obj = graph.value(cast(rdflib.URIRef, fact), RDF.object)
        if not isinstance(subject, rdflib.URIRef) or not isinstance(predicate, rdflib.URIRef):
            continue
        if not isinstance(obj, rdflib.URIRef):
            continue
        if predicate in (RDF.type, RDFS.label):
            continue
        subject_label = literal(graph, subject, RDFS.label) or local_name(str(subject))
        object_label = literal(graph, obj, RDFS.label) or local_name(str(obj))
        predicate_label = label_for_iri(graph, predicate) or local_name(str(predicate))
        rows.append(f"- {subject_label} - {predicate_label} -> {object_label}")
    if not rows:
        return []
    return ["## Extracted Relations", *sorted(set(rows)), ""]


def ontology_proposals_markdown(graph: rdflib.Graph, doc: rdflib.URIRef) -> list[str]:
    proposals = uris(graph, doc, ECHOG.hasOntologyProposal)
    if not proposals:
        return []
    lines = ["## Ontology Addition Proposals", ""]
    for proposal in sorted(
        proposals,
        key=lambda node: (
            literal(graph, node, ECHOG.proposalKind) or "",
            literal(graph, node, ECHOG.proposedLabel) or str(node),
        ),
    ):
        label = literal_any(graph, proposal, (ECHOG.proposedLabel, RDFS.label)) or local_name(
            str(proposal)
        )
        kind = literal(graph, proposal, ECHOG.proposalKind) or "term"
        reason = literal(graph, proposal, ECHOG.proposalReason)
        evidence_node = first_uri(graph, proposal, ECHO.hasEvidence)
        evidence = literal(graph, evidence_node, ECHO.sourceExcerpt) if evidence_node else None
        line = f"- `{kind}`: {label}"
        if reason:
            line += f" - {reason}"
        if evidence:
            line += f" Evidence: {evidence}"
        lines.append(line)
    lines.append("")
    return lines


def write_extraction_json(extraction: GenericExtraction, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(extraction), indent=2, ensure_ascii=False), encoding="utf-8")


def add_extraction_run(
    graph: rdflib.Graph,
    document: Document,
    *,
    model_name: str | None,
    extracted_at: datetime | None,
) -> rdflib.URIRef:
    run = ECHOG[f"run_{document.sha256[:8]}"]
    model_label = model_name or "unspecified local extractor"
    model = ECHOG[f"model_{slug(model_label)}"]
    prompt = ECHOG[f"prompt_{stable_suffix(system_prompt())}"]

    graph.add((run, RDF.type, ECHO.ExtractionRun))
    timestamp = extracted_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    timestamp = timestamp.replace(microsecond=0)
    graph.add((run, ECHO.extractedAt, rdflib.Literal(timestamp.isoformat(), datatype=XSD.dateTime)))
    graph.add((run, ECHO.usedModel, model))
    graph.add((run, ECHO.usedPrompt, prompt))

    graph.add((model, RDF.type, ECHO.ExtractionModel))
    graph.add((model, RDFS.label, rdflib.Literal(model_label)))
    graph.add((model, OWL.versionInfo, rdflib.Literal(model_label)))

    graph.add((prompt, RDF.type, ECHO.Prompt))
    graph.add((prompt, RDFS.label, rdflib.Literal("ECHO-G graph extraction prompt")))
    graph.add((prompt, ECHO.hasOriginalText, rdflib.Literal(system_prompt())))
    return run


def add_ontology_proposal(
    graph: rdflib.Graph,
    proposal: OntologyProposal,
    extraction: GenericExtraction,
    *,
    document_node: rdflib.URIRef,
    section_nodes: Mapping[int, rdflib.URIRef],
    run_node: rdflib.URIRef,
) -> rdflib.URIRef:
    node = ECHOG[f"ontology_proposal_{stable_suffix(str(document_node), proposal.proposal_id)}"]
    evidence = locate_evidence(extraction.sections, proposal.evidence or proposal.label)
    evidence_node = add_evidence(graph, evidence, document_node, section_nodes)

    graph.add((node, RDF.type, ECHOG.OntologyAdditionProposal))
    graph.add((node, RDF.type, ECHO.Annotation))
    graph.add((node, RDFS.label, rdflib.Literal(proposal.label)))
    graph.add((node, ECHOG.proposedLabel, rdflib.Literal(proposal.label)))
    graph.add((node, ECHOG.proposalKind, rdflib.Literal(proposal.kind)))
    graph.add((node, ECHO.hasEvidence, evidence_node))
    graph.add((node, ECHO.extractedBy, run_node))
    graph.add((document_node, ECHOG.hasOntologyProposal, node))
    graph.add((document_node, ECHO.mentions, node))
    if proposal.reason:
        graph.add((node, ECHOG.proposalReason, rdflib.Literal(proposal.reason)))
    for predicate, iri in (
        (ECHOG.suggestedParent, proposal.suggested_parent_iri),
        (ECHOG.suggestedDomain, proposal.suggested_domain_iri),
        (ECHOG.suggestedRange, proposal.suggested_range_iri),
    ):
        if iri:
            graph.add((node, predicate, rdflib.URIRef(iri)))
    return node


def add_assertion(
    graph: rdflib.Graph,
    subject: rdflib.URIRef,
    predicate: rdflib.URIRef,
    obj: Node,
    extraction: GenericExtraction,
    *,
    document_node: rdflib.URIRef,
    section_nodes: Mapping[int, rdflib.URIRef],
    run_node: rdflib.URIRef,
    evidence_text: str | None = None,
    confidence: float | None = None,
    certainty: str | None = None,
    polarity: str | None = None,
    unit_text: str | None = None,
) -> None:
    graph.add((subject, predicate, obj))
    evidence = locate_evidence(extraction.sections, evidence_text or object_text(obj))
    evidence_node = add_evidence(graph, evidence, document_node, section_nodes)
    fact = ECHOG[f"fact_{stable_suffix(str(subject), str(predicate), object_text(obj))}"]
    graph.add((fact, RDF.type, ECHO.ExtractedFact))
    graph.add((fact, RDF.subject, subject))
    graph.add((fact, RDF.predicate, predicate))
    graph.add((fact, RDF.object, obj))
    graph.add((fact, ECHO.hasSubject, subject))
    graph.add((fact, ECHO.hasEvidence, evidence_node))
    graph.add((fact, ECHO.extractedBy, run_node))
    graph.add((fact, ECHO.hasValidationStatus, ECHO.validationPending))
    graph.add((fact, ECHO.hasCertainty, certainty_concept(certainty)))
    graph.add((fact, ECHO.hasPolarity, polarity_concept(polarity)))
    graph.add(
        (
            fact,
            ECHO.confidence,
            rdflib.Literal(
                confidence if confidence is not None else DEFAULT_FACT_CONFIDENCE,
                datatype=XSD.decimal,
            ),
        )
    )
    graph.add((fact, PROV.wasDerivedFrom, evidence_node))
    graph.add((fact, PROV.wasGeneratedBy, run_node))

    if isinstance(obj, rdflib.URIRef):
        if predicate not in (RDF.type, RDFS.label):
            graph.add((fact, ECHO.hasRelation, predicate))
            graph.add((fact, ECHO.hasObject, obj))
        return

    if isinstance(obj, rdflib.Literal) and predicate != RDFS.label:
        variable = add_variable_node(graph, predicate)
        value = add_value_node(graph, predicate, str(obj), unit_text)
        graph.add((fact, ECHO.hasAttribute, variable))
        graph.add((fact, ECHO.hasValue, value))


def add_evidence(
    graph: rdflib.Graph,
    evidence: Evidence,
    document_node: rdflib.URIRef,
    section_nodes: Mapping[int, rdflib.URIRef],
) -> rdflib.URIRef:
    node = ECHOG[
        f"evidence_{stable_suffix(evidence.heading, str(evidence.page_number), evidence.excerpt)}"
    ]
    if (node, RDF.type, ECHO.Evidence) in graph:
        return node
    graph.add((node, RDF.type, ECHO.Evidence))
    graph.add((node, RDF.type, ECHO.TextSpan))
    graph.add((node, RDFS.label, rdflib.Literal(evidence.excerpt[:80])))
    graph.add((node, ECHO.sourceExcerpt, rdflib.Literal(evidence.excerpt)))
    graph.add(
        (node, ECHO.hasOriginalText, rdflib.Literal(evidence.original_text or evidence.excerpt))
    )
    graph.add((node, ECHO.sourcePage, rdflib.Literal(evidence.page_number, datatype=XSD.integer)))
    graph.add((node, ECHO.charStart, rdflib.Literal(evidence.char_start, datatype=XSD.integer)))
    graph.add((node, ECHO.charEnd, rdflib.Literal(evidence.char_end, datatype=XSD.integer)))
    graph.add((node, ECHO.sourceDocument, document_node))
    section_node = section_nodes.get(evidence.section_order)
    if section_node is not None:
        graph.add((node, ECHO.sourceSection, section_node))
    return node


def locate_evidence(sections: tuple[SourceSection, ...], value: str) -> Evidence:
    needle = compact(value).casefold()
    for section in sections:
        if not needle or needle in compact(section.text).casefold():
            start, end, original = locate_original_span(section.text, value)
            return Evidence(
                section.heading,
                section.page_number,
                excerpt(section.text, value),
                section.order,
                start,
                end,
                original,
            )
    if sections:
        section = sections[0]
        start, end, original = locate_original_span(section.text, value)
        return Evidence(
            section.heading,
            section.page_number,
            excerpt(section.text, value),
            section.order,
            start,
            end,
            original,
        )
    return Evidence("Document", 1, value[:240], 1, 0, len(value), value[:240])


def add_variable_node(graph: rdflib.Graph, predicate: rdflib.URIRef) -> rdflib.URIRef:
    node = ECHOG[f"variable_{stable_suffix(str(predicate))}"]
    if (node, RDF.type, ECHO.Variable) in graph:
        return node
    graph.add((node, RDF.type, ECHO.Variable))
    graph.add((node, RDFS.label, rdflib.Literal(local_name(str(predicate)))))
    graph.add((node, ECHO.wasDerivedFrom, predicate))
    return node


def add_value_node(
    graph: rdflib.Graph,
    predicate: rdflib.URIRef,
    value: str,
    unit_text: str | None,
) -> rdflib.URIRef:
    node = ECHOG[f"value_{stable_suffix(str(predicate), value, unit_text or '')}"]
    if (node, RDF.type, ECHO.Value) in graph:
        return node
    graph.add((node, RDF.type, ECHO.Value))
    graph.add((node, RDFS.label, rdflib.Literal(value)))
    numeric_value = normalized_numeric_literal(value)
    if numeric_value is not None:
        graph.add((node, RDF.type, ECHO.Quantity if unit_text else ECHO.NumericValue))
        graph.add((node, ECHO.numericValue, rdflib.Literal(numeric_value, datatype=XSD.decimal)))
    else:
        graph.add((node, RDF.type, ECHO.CategoricalValue))
        graph.add((node, ECHO.categoricalValue, rdflib.Literal(value)))
    if unit_text:
        unit = ECHOG[f"unit_{slug(unit_text)}"]
        graph.add((unit, RDF.type, ECHO.Unit))
        graph.add((unit, RDFS.label, rdflib.Literal(unit_text)))
        graph.add((node, ECHO.hasUnit, unit))
    return node


def locate_original_span(source: str, value: str) -> tuple[int, int, str]:
    needle = compact(value)
    if not source:
        return 0, 0, ""
    if not needle:
        return 0, min(len(source), 240), source[:240]
    index = source.casefold().find(needle.casefold())
    if index != -1:
        end = index + len(needle)
        return index, end, source[index:end]
    tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{2,}", needle))
    for token in sorted(tokens, key=len, reverse=True):
        index = source.casefold().find(token.casefold())
        if index != -1:
            end = min(len(source), index + max(len(token), min(len(needle), 240)))
            return index, end, source[index:end]
    end = min(len(source), max(len(needle), 1))
    return 0, end, source[:end]


def normalized_numeric_literal(value: str) -> str | None:
    text = compact(value).replace(",", ".")
    if NUMERIC_LITERAL.match(text):
        return text
    return None


def certainty_concept(certainty: str | None) -> rdflib.URIRef:
    key = compact(certainty or "").casefold().replace("-", "_")
    return {
        "certain": ECHO.certain,
        "probable": ECHO.probable,
        "possible": ECHO.possible,
        "unverified": ECHO.unverified,
        "human_confirmed": ECHO.humanConfirmed,
    }.get(key, ECHO.unverified)


def polarity_concept(polarity: str | None) -> rdflib.URIRef:
    key = compact(polarity or "").casefold()
    return {
        "affirmed": ECHO.polarityAffirmed,
        "positive": ECHO.polarityAffirmed,
        "+1": ECHO.polarityAffirmed,
        "negated": ECHO.polarityNegated,
        "negative": ECHO.polarityNegated,
        "-1": ECHO.polarityNegated,
    }.get(key, ECHO.polarityAffirmed)


def select_document(graph: rdflib.Graph, document_iri: str | None) -> rdflib.URIRef:
    if document_iri:
        doc = rdflib.URIRef(document_iri)
        if (doc, RDF.type, ECHOG.ProcessedDocument) not in graph and (
            doc,
            RDF.type,
            ECHO.Document,
        ) not in graph:
            raise EchoGError(f"IRI is not an ECHO-G processed document: {document_iri}")
        return doc
    docs = sorted(
        node
        for node in graph.subjects(RDF.type, ECHOG.ProcessedDocument)
        if isinstance(node, rdflib.URIRef)
    )
    if not docs:
        docs = sorted(
            node
            for node in graph.subjects(RDF.type, ECHO.Document)
            if isinstance(node, rdflib.URIRef)
        )
    if len(docs) != 1:
        raise EchoGError(f"expected one processed document, found {len(docs)}")
    return docs[0]


def first_non_core_type(graph: rdflib.Graph, node: rdflib.URIRef) -> rdflib.URIRef | None:
    for obj in graph.objects(node, RDF.type):
        if isinstance(obj, rdflib.URIRef) and not str(obj).startswith(str(ECHOG)):
            return obj
    return None


def literal_attributes(
    graph: rdflib.Graph, entity: rdflib.URIRef
) -> tuple[tuple[rdflib.URIRef, str], ...]:
    skip = {
        RDFS.label,
        ECHO.hasOriginalText,
        ECHO.hasText,
        ECHO.hasTitle,
        ECHO.hasHeading,
        ECHO.hasOrder,
        ECHO.hasPageNumber,
        ECHO.hasSourceFile,
        ECHO.hasSha256,
        ECHO.sourceExcerpt,
        ECHO.sourcePage,
        ECHO.charStart,
        ECHO.charEnd,
        ECHO.confidence,
        ECHO.extractedAt,
    }
    values: list[tuple[rdflib.URIRef, str]] = []
    for predicate, obj in graph.predicate_objects(entity):
        if predicate in skip or predicate == RDF.type:
            continue
        if str(predicate).startswith(str(ECHOG)):
            continue
        if isinstance(predicate, rdflib.URIRef) and isinstance(obj, rdflib.Literal):
            values.append((predicate, str(obj)))
    return tuple(sorted(values, key=lambda row: (str(row[0]), row[1])))


def extracted_fact_nodes(graph: rdflib.Graph) -> tuple[rdflib.URIRef, ...]:
    nodes = {
        node
        for rdf_type in (ECHO.ExtractedFact, ECHOG.ExtractedFact)
        for node in graph.subjects(RDF.type, rdf_type)
        if isinstance(node, rdflib.URIRef)
    }
    return tuple(sorted(nodes))


def first_uri(
    graph: rdflib.Graph, subject: rdflib.URIRef, predicate: rdflib.URIRef
) -> rdflib.URIRef | None:
    values = uris(graph, subject, predicate)
    return values[0] if values else None


def uris(
    graph: rdflib.Graph, subject: rdflib.URIRef, predicate: rdflib.URIRef
) -> tuple[rdflib.URIRef, ...]:
    return tuple(obj for obj in graph.objects(subject, predicate) if isinstance(obj, rdflib.URIRef))


def uris_any(
    graph: rdflib.Graph,
    subject: rdflib.URIRef,
    predicates: tuple[rdflib.URIRef, ...],
) -> tuple[rdflib.URIRef, ...]:
    values: list[rdflib.URIRef] = []
    for predicate in predicates:
        values.extend(uris(graph, subject, predicate))
    return tuple(dict.fromkeys(values))


def literal(
    graph: rdflib.Graph, subject: rdflib.URIRef | None, predicate: rdflib.URIRef
) -> str | None:
    if subject is None:
        return None
    values = sorted(
        str(obj) for obj in graph.objects(subject, predicate) if isinstance(obj, rdflib.Literal)
    )
    return values[0] if values else None


def literal_any(
    graph: rdflib.Graph,
    subject: rdflib.URIRef | None,
    predicates: tuple[rdflib.URIRef, ...],
) -> str | None:
    for predicate in predicates:
        value = literal(graph, subject, predicate)
        if value is not None:
            return value
    return None


def label_for_iri(graph: rdflib.Graph, iri: rdflib.URIRef | None) -> str | None:
    if iri is None:
        return None
    return literal(graph, iri, RDFS.label) or local_name(str(iri))


def bind_namespaces(graph: rdflib.Graph) -> None:
    graph.bind("echo", ECHO)
    graph.bind("echog", ECHOG)
    graph.bind("dcterms", DCTERMS)
    graph.bind("prov", PROV)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)


def add_core_vocabulary(graph: rdflib.Graph) -> None:
    labels = {
        ECHOG.ProcessedDocument: "Processed document",
        ECHOG.OntologyAdditionProposal: "Ontology addition proposal",
        ECHO.Document: "Document",
        ECHO.DocumentSection: "Document section",
        ECHO.Evidence: "Evidence",
        ECHO.TextSpan: "Text span",
        ECHO.ExtractedFact: "Extracted fact",
        ECHO.ExtractionRun: "Extraction run",
        ECHO.ExtractionModel: "Extraction model",
        ECHO.Prompt: "Prompt",
        ECHO.Variable: "Variable",
        ECHO.Value: "Value",
        ECHO.Quantity: "Quantity",
        ECHO.NumericValue: "Numeric value",
        ECHO.CategoricalValue: "Categorical value",
        ECHO.Unit: "Unit",
    }
    for node, label in labels.items():
        graph.add((node, RDF.type, OWL.Class))
        graph.add((node, RDFS.label, rdflib.Literal(label)))
    graph.add((ECHOG.ProcessedDocument, RDFS.subClassOf, ECHO.Document))
    for predicate in (
        ECHO.hasSection,
        ECHO.hasEntity,
        ECHO.mentions,
        ECHO.hasEvidence,
        ECHO.sourceSection,
        ECHO.sourceDocument,
        ECHO.hasSubject,
        ECHO.hasObject,
        ECHO.hasAttribute,
        ECHO.hasValue,
        ECHO.hasUnit,
        ECHO.hasRelation,
        ECHO.extractedBy,
        ECHO.usedModel,
        ECHO.usedPrompt,
        ECHO.hasValidationStatus,
        ECHO.hasCertainty,
        ECHO.hasPolarity,
        ECHO.wasDerivedFrom,
        ECHOG.hasOntologyProposal,
        ECHOG.suggestedParent,
        ECHOG.suggestedDomain,
        ECHOG.suggestedRange,
    ):
        graph.add((predicate, RDF.type, OWL.ObjectProperty))
    for predicate in (
        ECHO.hasTitle,
        ECHO.hasHeading,
        ECHO.hasText,
        ECHO.hasOriginalText,
        ECHO.hasOrder,
        ECHO.hasPageNumber,
        ECHO.hasSha256,
        ECHO.hasSourceFile,
        ECHO.sourceExcerpt,
        ECHO.sourcePage,
        ECHO.charStart,
        ECHO.charEnd,
        ECHO.numericValue,
        ECHO.categoricalValue,
        ECHO.confidence,
        ECHO.extractedAt,
        ECHOG.proposedLabel,
        ECHOG.proposalKind,
        ECHOG.proposalReason,
    ):
        graph.add((predicate, RDF.type, OWL.DatatypeProperty))


def _terms(graph: rdflib.Graph, rdf_type: rdflib.URIRef, kind: str) -> Iterable[TBoxTerm]:
    terms: list[TBoxTerm] = []
    for subject in graph.subjects(RDF.type, rdf_type):
        if not isinstance(subject, rdflib.URIRef):
            continue
        terms.append(
            TBoxTerm(
                iri=str(subject),
                label=literal(graph, subject, RDFS.label) or local_name(str(subject)),
                kind=kind,
                domain=_first_object_iri(graph, subject, RDFS.domain),
                range=_first_object_iri(graph, subject, RDFS.range),
                comment=literal(graph, subject, RDFS.comment),
            )
        )
    return sorted(terms, key=lambda term: (term.label.casefold(), term.iri))


def _first_object_iri(
    graph: rdflib.Graph, subject: rdflib.URIRef, predicate: rdflib.URIRef
) -> str | None:
    obj = next(
        (item for item in graph.objects(subject, predicate) if isinstance(item, rdflib.URIRef)),
        None,
    )
    return str(obj) if obj is not None else None


def _term_line(term: TBoxTerm) -> str:
    pieces = [f"- <{term.iri}> label={term.label!r}"]
    if term.domain:
        pieces.append(f"domain=<{term.domain}>")
    if term.range:
        pieces.append(f"range=<{term.range}>")
    if term.comment:
        pieces.append(f"comment={term.comment[:120]!r}")
    return " ".join(pieces)


def list_of_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(cast(Mapping[str, object], item) for item in value if isinstance(item, dict))


def string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = compact(value)
        return text or None
    if isinstance(value, int | float):
        return str(value)
    return None


def float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    if 0.0 <= number <= 1.0:
        return number
    return None


def iri_or_none(value: object) -> str | None:
    text = string_or_none(value)
    if text is None:
        return None
    if text.startswith("<") and text.endswith(">"):
        return text[1:-1].strip() or None
    return text


def clean_multiline(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def humanize_label(value: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value.replace("_", " ").replace("-", " "))
    return compact(separated)


def looks_like_instance_label(label: str) -> bool:
    text = humanize_label(compact(label))
    if not text:
        return False
    if PERSON_TITLE.match(text):
        return True
    return bool(IDENTIFIED_INSTANCE.match(text))


def excerpt(text: str, value: str) -> str:
    source = compact(text)
    needle = compact(value)
    if not source:
        return needle[:240]
    if not needle:
        return source[:240]
    index = source.casefold().find(needle.casefold())
    if index == -1:
        return source[:240]
    start = max(0, index - 80)
    end = min(len(source), index + len(needle) + 80)
    return source[start:end]


def stable_suffix(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def slug(text: str) -> str:
    normalized = text.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return value or stable_suffix(text)


def local_name(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            return iri.rsplit(sep, 1)[-1]
    return iri


def object_text(obj: Node) -> str:
    return str(obj)

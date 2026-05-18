"""End-to-end evaluation of the PDF -> MD -> TTL -> MD' pipeline.

Runs the full round-trip for each PDF and reports semantic idempotence metrics
(recall, precision, hallucination rate) per document and an aggregate summary.

Resume behaviour (default):
    A PDF is skipped if its report already exists and is complete.
    Re-run the script on the same --out directory to continue after an interruption.

Force re-run:
    Pass --force to erase existing results and reprocess every PDF.

Usage — folder mode (recommended):
    python evaluate_pipeline.py --input-dir eval/sources/ \\
        --tbox ontology/echo-core.ttl --out eval/runs/run_001

Usage — explicit files:
    python evaluate_pipeline.py doc1.pdf doc2.pdf \\
        --tbox ontology/echo-core.ttl --out eval/runs/run_001

Usage — force reprocess:
    python evaluate_pipeline.py --input-dir eval/sources/ \\
        --tbox ontology/echo-core.ttl --out eval/runs/run_001 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import ollama

from compare_semantics import DEFAULT_MATCH_THRESHOLD, EMBED_MODEL, run as compare_run
from echo_g_core import DEFAULT_HOST, DEFAULT_MODEL
from ontology_to_markdown import run as downlift_run
from pdf_to_markdown import run as pdf_to_md_run
from pdf_to_ontology import run as uplift_run

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ──────────────────────────────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _print(msg: str) -> None:
    print(msg, flush=True)


class _LiveStep:
    """Prints a live elapsed-time ticker on stdout while a step runs.

    In TTY mode the line is overwritten every two seconds so the terminal
    stays tidy.  In non-TTY mode a single start line is printed instead.
    """

    _INTERVAL = 2.0

    def __init__(self, label: str) -> None:
        self._label = label
        self._start = time.time()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._tick, daemon=True)
        self.status: str = ""  # set from outside (e.g. chunk progress callback)

    def __enter__(self) -> "_LiveStep":
        if _IS_TTY:
            self._t.start()
        else:
            print(f"    {self._label} ...", flush=True)
        return self

    def __exit__(self, *_: object) -> None:
        if _IS_TTY:
            self._stop.set()
            self._t.join()
            print(" " * 80, end="\r", flush=True)

    def _tick(self) -> None:
        while not self._stop.wait(timeout=self._INTERVAL):
            elapsed = time.time() - self._start
            status = f"  {self.status}" if self.status else ""
            print(
                f"    {self._label:<38}  {elapsed:7.1f}s{status}",
                end="\r", flush=True,
            )

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


class _WarnCounter(logging.Handler):
    """Counts WARNING records emitted to any logger during a step."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING:
            self.count += 1


@contextmanager
def _step(label: str) -> Generator[tuple["_LiveStep", "_WarnCounter"], None, None]:
    """Combined context manager: live ticker + warning counter for one step."""
    counter = _WarnCounter()
    logging.getLogger().addHandler(counter)
    try:
        with _LiveStep(label) as live:
            yield live, counter
    finally:
        logging.getLogger().removeHandler(counter)


def _step_done(
    label: str, elapsed: float, *, info: str = "", warn: int = 0
) -> None:
    warn_tag = f"  [{warn} warn]" if warn else ""
    info_col = f"  {info}" if info else ""
    print(f"    {label:<38}{info_col:<30} {elapsed:7.1f}s{warn_tag}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model version helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_model_info(host: str, model_name: str) -> dict[str, object]:
    """Query Ollama for model details: digest, family, size, quantization.

    For MoE models, ``parameter_size`` (from details) reflects active params
    while ``total_parameters`` (from model_info) reflects all weights.  Both
    are stored when available so the distinction is visible in the report.
    """
    try:
        client = ollama.Client(host=host)
        info = client.show(model_name)
        result: dict[str, object] = {"name": model_name}
        if info.details:
            d = info.details
            for key, val in [
                ("architecture", d.family),
                ("quantization", d.quantization_level),
                ("format", d.format),
            ]:
                if val:
                    result[key] = val
            # parameter_size from details = active params for MoE models
            if d.parameter_size:
                result["active_parameters"] = d.parameter_size

        # model_info dict carries total parameter count (authoritative)
        raw_info: dict = getattr(info, "model_info", None) or {}
        total = raw_info.get("general.parameter_count")
        if total:
            # format as human-readable string, e.g. 36.0B
            total_b = int(total) / 1e9
            result["total_parameters"] = f"{total_b:.1f}B"

        # Resolve the short content digest from the model list
        try:
            for m in client.list().models:
                tag = getattr(m, "model", None) or getattr(m, "name", None)
                if tag == model_name and m.digest:
                    result["digest"] = m.digest[:12]
                    break
        except Exception:
            pass
        return result
    except Exception as exc:
        logger.warning("Could not fetch info for model %s: %s", model_name, exc)
        return {"name": model_name, "error": str(exc)}


def _models_path(out_dir: Path) -> Path:
    return out_dir / "models.json"


def _model_tag(info: dict) -> str:
    parts = []
    total = info.get("total_parameters")
    active = info.get("active_parameters")
    if total and active:
        parts.append(f"{total} total / {active} active")
    elif total:
        parts.append(total)
    elif active:
        parts.append(active)
    if info.get("quantization"):
        parts.append(info["quantization"])
    return f"  [{', '.join(parts)}]" if parts else ""


def collect_models_info(
    host: str, model: str, embed_model: str, out_dir: Path
) -> dict[str, object]:
    """Return version details for every LLM used in the pipeline.

    If ``out_dir/models.json`` already exists the info is loaded from disk
    (no Ollama query needed — useful when resuming an interrupted run).
    Otherwise Ollama is queried and the result is written to that file.
    """
    mp = _models_path(out_dir)
    if mp.exists():
        try:
            info = json.loads(mp.read_text(encoding="utf-8"))
            _print(f"  Model info loaded from {mp.name} (cached).")
            _print(f"  uplift / compare : {info['uplift_model'].get('name', model)}{_model_tag(info['uplift_model'])}")
            _print(f"  embedding        : {info['embed_model'].get('name', embed_model)}{_model_tag(info['embed_model'])}")
            _print("")
            return info
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt %s — re-querying Ollama.", mp)

    _print("Fetching model info from Ollama ...")
    uplift_info = get_model_info(host, model)
    embed_info = get_model_info(host, embed_model)
    _print(f"  uplift / compare : {model}{_model_tag(uplift_info)}")
    _print(f"  embedding        : {embed_model}{_model_tag(embed_info)}")
    _print("")

    info = {"uplift_model": uplift_info, "embed_model": embed_info}
    mp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Resume / cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _report_path(name: str, out_dir: Path) -> Path:
    return out_dir / "reports" / f"{name}.json"


def is_done(name: str, out_dir: Path) -> bool:
    """Return True if the PDF was already processed successfully."""
    rp = _report_path(name, out_dir)
    if not rp.exists():
        return False
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        return "summary" in data and "error" not in data
    except (json.JSONDecodeError, OSError):
        return False


def load_cached_result(name: str, pdf_path: Path, out_dir: Path) -> dict[str, object]:
    """Reconstruct a result dict from an existing report file."""
    rp = _report_path(name, out_dir)
    data = json.loads(rp.read_text(encoding="utf-8"))
    return {
        "name": name,
        "pdf": str(pdf_path.resolve()),
        "markdown": str(out_dir / "markdown" / f"{name}.md"),
        "ttl": str(out_dir / "ttl" / f"{name}.ttl"),
        "downlift": str(out_dir / "downlift" / f"{name}.md"),
        "report": str(rp.resolve()),
        "timings_s": {"cached": True},
        "uplift": {},
        "metrics": data["summary"],
        "cached": True,
    }


def clear_results(name: str, out_dir: Path) -> None:
    """Delete all output files for a single PDF so it can be reprocessed."""
    targets = [
        out_dir / "markdown" / f"{name}.md",
        out_dir / "ttl" / f"{name}.ttl",
        out_dir / "ttl" / f"{name}.json",
        out_dir / "downlift" / f"{name}.md",
        out_dir / "reports" / f"{name}.json",
    ]
    for p in targets:
        if p.exists():
            p.unlink()
            logger.debug("Deleted %s", p)


# ──────────────────────────────────────────────────────────────────────────────
# Per-document evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_one(
    pdf_path: Path,
    *,
    out_dir: Path,
    tbox: Path,
    model: str,
    host: str,
    embed_model: str,
    threshold: float,
    max_attempts: int,
) -> dict[str, object]:
    name = pdf_path.stem
    md_path = out_dir / "markdown" / f"{name}.md"
    ttl_path = out_dir / "ttl" / f"{name}.ttl"
    json_path = out_dir / "ttl" / f"{name}.json"
    downlift_path = out_dir / "downlift" / f"{name}.md"
    report_path = out_dir / "reports" / f"{name}.json"

    with _step("[1/4] PDF -> Markdown") as (live, warns):
        pdf_to_md_run(pdf_path, md_path)
    t_pdf_md = live.elapsed
    _step_done("[1/4] PDF -> Markdown", t_pdf_md, warn=warns.count)

    with _step("[2/4] Uplift") as (live, warns):
        def _on_chunk(idx: int, total: int) -> None:
            live.status = f"chunk {idx}/{total}"

        uplift_result = uplift_run(
            source_path=md_path,
            tbox_path=tbox,
            output_path=ttl_path,
            json_output_path=json_path,
            content_dir=out_dir / "content_store",
            model=model,
            host=host,
            max_attempts=max_attempts,
            max_tbox_terms=400,
            domain_hint="",
            progress_callback=_on_chunk,
        )
    t_uplift = live.elapsed
    tc = uplift_result.get("triple_count") or 0
    fc = uplift_result.get("fact_count") or 0
    ec = uplift_result.get("entity_count") or 0
    _step_done(
        "[2/4] Uplift", t_uplift,
        info=f"{tc} triples | {ec} entities | {fc} facts",
        warn=warns.count,
    )

    with _step("[3/4] Downlift") as (live, warns):
        downlift_run(
            ontology_path=ttl_path,
            output_path=downlift_path,
            document_iri=None,
        )
    t_downlift = live.elapsed
    _step_done("[3/4] Downlift", t_downlift, warn=warns.count)

    with _step("[4/4] Semantic compare") as (live, warns):
        comparison = compare_run(
            md_path,
            downlift_path,
            output_path=report_path,
            host=host,
            model=model,
            embed_model=embed_model,
            threshold=threshold,
        )
    t_compare = live.elapsed
    m = comparison["summary"]
    fo = m.get("facts_original", 0)
    fr = m.get("facts_reconstructed", 0)
    _step_done(
        "[4/4] Semantic compare", t_compare,
        info=f"orig={fo} | recon={fr} facts",
        warn=warns.count,
    )

    _print(
        f"    => recall={m['recall']:.3f}  "
        f"precision={m['precision']:.3f}  "
        f"halluc={m['hallucination_rate']:.3f}  "
        f"sim={m['mean_match_similarity']:.3f}"
    )

    return {
        "name": name,
        "pdf": str(pdf_path.resolve()),
        "markdown": str(md_path.resolve()),
        "ttl": str(ttl_path.resolve()),
        "downlift": str(downlift_path.resolve()),
        "report": str(report_path.resolve()),
        "timings_s": {
            "pdf_to_md": round(t_pdf_md, 2),
            "uplift": round(t_uplift, 2),
            "downlift": round(t_downlift, 2),
            "compare": round(t_compare, 2),
        },
        "uplift": {
            "triple_count": uplift_result.get("triple_count"),
            "entity_count": uplift_result.get("entity_count"),
            "relation_count": uplift_result.get("relation_count"),
            "fact_count": uplift_result.get("fact_count"),
            "proposal_count": uplift_result.get("proposal_count"),
        },
        "metrics": m,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run(
    pdf_paths: list[Path],
    *,
    out_dir: Path,
    tbox: Path,
    model: str,
    host: str,
    embed_model: str,
    threshold: float,
    max_attempts: int,
    force: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(pdf_paths)

    _print(f"\nEvaluating {total} PDF(s)  [out: {out_dir}]")
    if force:
        _print("  --force: existing results will be erased and reprocessed.")
    _print("")

    models_info = collect_models_info(host, model, embed_model, out_dir)

    results: list[dict[str, object]] = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        name = pdf_path.stem
        _print(f"{'=' * 60}")
        _print(f"[{idx}/{total}] {pdf_path.name}")

        if force:
            clear_results(name, out_dir)
        elif is_done(name, out_dir):
            cached = load_cached_result(name, pdf_path, out_dir)
            m = cached["metrics"]
            _print(
                f"    (cached) recall={m['recall']:.3f}  "
                f"precision={m['precision']:.3f}  "
                f"halluc={m['hallucination_rate']:.3f}  "
                f"sim={m['mean_match_similarity']:.3f}"
            )
            results.append(cached)
            continue

        try:
            result = evaluate_one(
                pdf_path,
                out_dir=out_dir,
                tbox=tbox,
                model=model,
                host=host,
                embed_model=embed_model,
                threshold=threshold,
                max_attempts=max_attempts,
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Evaluation failed for %s", pdf_path)
            _print(f"    ERROR: {exc}")
            results.append({"name": name, "error": str(exc)})

    _print(f"{'=' * 60}\n")

    metrics = [r["metrics"] for r in results if "metrics" in r]
    aggregate = None
    if metrics:
        aggregate = {
            "documents": len(metrics),
            "mean_recall": round(sum(m["recall"] for m in metrics) / len(metrics), 4),
            "mean_precision": round(sum(m["precision"] for m in metrics) / len(metrics), 4),
            "mean_hallucination_rate": round(
                sum(m["hallucination_rate"] for m in metrics) / len(metrics), 4
            ),
            "mean_match_similarity": round(
                sum(m["mean_match_similarity"] for m in metrics) / len(metrics), 4
            ),
            "total_original_facts": sum(m["facts_original"] for m in metrics),
            "total_reconstructed_facts": sum(m["facts_reconstructed"] for m in metrics),
            "total_missing": sum(m["missing_count"] for m in metrics),
            "total_hallucinated": sum(m["hallucinated_count"] for m in metrics),
        }

    summary_path = out_dir / "summary.json"
    summary = {
        "models": models_info,
        "threshold": threshold,
        "results": results,
        "aggregate": aggregate,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation.")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dir",
        type=Path,
        metavar="DIR",
        help="Folder containing PDF files to evaluate (all *.pdf files are processed).",
    )
    src.add_argument(
        "pdfs",
        type=Path,
        nargs="*",
        default=[],
        metavar="PDF",
        help="One or more PDF files to evaluate (alternative to --input-dir).",
    )

    parser.add_argument("--tbox", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Erase existing results and reprocess every PDF from scratch.",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def print_aggregate(summary: dict[str, object]) -> None:
    models = summary.get("models", {})
    uplift = models.get("uplift_model", {})
    embed = models.get("embed_model", {})

    agg = summary.get("aggregate")
    print("\n" + "=" * 72)
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * 72)

    # Model version block
    def _fmt(info: dict) -> str:
        parts = []
        total = info.get("total_parameters")
        active = info.get("active_parameters")
        if total and active:
            parts.append(f"{total} total / {active} active")
        elif total:
            parts.append(total)
        elif active:
            parts.append(active)
        if info.get("quantization"):
            parts.append(info["quantization"])
        if info.get("digest"):
            parts.append(f"digest={info['digest']}")
        return f"  [{', '.join(parts)}]" if parts else ""

    print(f"  uplift / compare : {uplift.get('name', '?')}{_fmt(uplift)}")
    print(f"  embedding        : {embed.get('name', '?')}{_fmt(embed)}")
    print("-" * 72)

    for r in summary["results"]:
        if "error" in r:
            print(f"  {r['name']:<34}  ERROR: {r['error']}")
            continue
        m = r["metrics"]
        cached_tag = "  (cached)" if r.get("cached") else ""
        print(
            f"  {r['name']:<34}"
            f"recall={m['recall']:.3f}  "
            f"precision={m['precision']:.3f}  "
            f"halluc={m['hallucination_rate']:.3f}  "
            f"sim={m['mean_match_similarity']:.3f}"
            f"{cached_tag}"
        )
    if agg:
        print("-" * 72)
        print(
            f"  {'AGGREGATE':<34}"
            f"recall={agg['mean_recall']:.3f}  "
            f"precision={agg['mean_precision']:.3f}  "
            f"halluc={agg['mean_hallucination_rate']:.3f}  "
            f"sim={agg['mean_match_similarity']:.3f}"
        )
    print("=" * 72 + "\n")


def resolve_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.input_dir:
        pdfs = sorted(args.input_dir.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDF files found in {args.input_dir}")
        return pdfs
    if not args.pdfs:
        raise SystemExit("Provide either --input-dir or at least one PDF file.")
    return list(args.pdfs)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pdf_paths = resolve_pdfs(args)
    summary = run(
        pdf_paths,
        out_dir=args.out,
        tbox=args.tbox,
        model=args.model,
        host=args.host,
        embed_model=args.embed_model,
        threshold=args.threshold,
        max_attempts=args.max_attempts,
        force=args.force,
    )
    print_aggregate(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

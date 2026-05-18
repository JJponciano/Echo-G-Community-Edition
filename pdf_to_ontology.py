"""CLI: ECHO-G generic source document + TBox to Turtle ontology."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from echo_g_core import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    GenericExtractor,
    LocalOllama,
    archive_source,
    build_ontology,
    ingest_source,
    load_tbox,
    write_extraction_json,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONTENT_DIR = APP_DIR / "content_store"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"


def run(
    *,
    source_path: Path,
    tbox_path: Path,
    output_path: Path | None,
    json_output_path: Path | None,
    content_dir: Path,
    model: str,
    host: str,
    max_attempts: int,
    max_tbox_terms: int,
    domain_hint: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    document, pages = ingest_source(source_path)
    archived_path = archive_source(source_path, content_dir, document.sha256)
    tbox = load_tbox(tbox_path, max_terms=max_tbox_terms)
    extraction = GenericExtractor(
        LocalOllama(model=model, host=host),
        max_attempts=max_attempts,
    ).extract(pages=pages, tbox=tbox, domain_hint=domain_hint,
              progress_callback=progress_callback)
    graph, report = build_ontology(
        document=document,
        extraction=extraction,
        tbox=tbox,
        model_name=model,
    )

    output = output_path or DEFAULT_OUTPUT_DIR / f"{source_path.stem}_{document.sha256[:16]}.ttl"
    json_output = json_output_path or output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    write_extraction_json(extraction, json_output)

    return {
        "approach": "ECHO-G",
        "source_document": str(source_path.resolve()),
        "source_format": document.source_format,
        "tbox": str(tbox_path.resolve()),
        "sha256": document.sha256,
        "archived_content": str(archived_path.resolve()),
        "ontology": str(output.resolve()),
        "json": str(json_output.resolve()),
        "document_iri": report.document_iri,
        "triple_count": report.triple_count,
        "entity_count": report.entity_count,
        "relation_count": report.relation_count,
        "fact_count": report.fact_count,
        "proposal_count": report.proposal_count,
        "model": model,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ECHO-G generic PDF/text/Markdown + required TBox to Turtle ontology."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--tbox",
        type=Path,
        required=True,
        help="Required domain TBox. ECHO-G has no built-in domain ontology.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--content-dir", type=Path, default=DEFAULT_CONTENT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tbox-terms", type=int, default=200)
    parser.add_argument(
        "--domain-hint",
        default="",
        help="Optional natural-language hint; does not replace the required TBox.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(
        json.dumps(
            run(
                source_path=args.source,
                tbox_path=args.tbox,
                output_path=args.output,
                json_output_path=args.json_output,
                content_dir=args.content_dir,
                model=args.model,
                host=args.host,
                max_attempts=args.max_attempts,
                max_tbox_terms=args.max_tbox_terms,
                domain_hint=args.domain_hint,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

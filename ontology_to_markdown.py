"""CLI: ECHO-G Turtle ontology to generic Markdown reconstruction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rdflib

from echo_g_core import reconstruct_markdown


def run(
    *,
    ontology_path: Path,
    output_path: Path | None,
    document_iri: str | None,
) -> dict[str, object]:
    graph = rdflib.Graph()
    graph.parse(source=str(ontology_path), format="turtle")
    markdown = reconstruct_markdown(graph, document_iri=document_iri)
    output = output_path or ontology_path.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return {
        "approach": "ECHO-G",
        "ontology": str(ontology_path.resolve()),
        "markdown": str(output.resolve()),
        "chars": len(markdown),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECHO-G Turtle ontology to Markdown.")
    parser.add_argument("ontology", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--document-iri", default=None)
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
                ontology_path=args.ontology,
                output_path=args.output,
                document_iri=args.document_iri,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

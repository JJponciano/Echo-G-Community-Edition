"""CLI: PDF -> Markdown extraction (pre-uplift step).

Produces a clean Markdown rendering of a PDF that preserves headings, page
breaks, and paragraph structure. Output is the canonical "source of truth"
that the uplift -> downlift round-trip is evaluated against.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-–—:/&.()]{2,80}$")
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+[A-Z].{1,120}$")
BULLET_RE = re.compile(r"^[•◦⁃\-\*]\s+")


def extract_pages(pdf_path: Path) -> list[str]:
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    if NUMBERED_HEADING_RE.match(stripped):
        return True
    if HEADING_RE.match(stripped) and len(stripped.split()) <= 12:
        letters = [c for c in stripped if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            return True
    return False


def to_markdown(pages: list[str], title: str) -> str:
    lines: list[str] = [f"# {title}", ""]
    for page_num, page_text in enumerate(pages, start=1):
        lines.append(f"<!-- page {page_num} -->")
        prev_blank = True
        for raw_line in page_text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if not prev_blank:
                    lines.append("")
                prev_blank = True
                continue
            prev_blank = False
            if looks_like_heading(line):
                lines.append("")
                lines.append(f"## {line.strip()}")
                lines.append("")
                continue
            if BULLET_RE.match(line.strip()):
                lines.append(f"- {BULLET_RE.sub('', line.strip(), count=1)}")
                continue
            lines.append(line)
        lines.append("")
    out = "\n".join(lines)
    # collapse 3+ blank lines to 2
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def run(pdf_path: Path, output_path: Path | None = None) -> dict[str, object]:
    pages = extract_pages(pdf_path)
    title = pdf_path.stem.replace("_", " ").replace("-", " ")
    md = to_markdown(pages, title)
    output_path = output_path or pdf_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return {
        "approach": "pdf-to-markdown",
        "pdf": str(pdf_path.resolve()),
        "markdown": str(output_path.resolve()),
        "pages": len(pages),
        "chars": len(md),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a PDF to clean Markdown.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(args.pdf, args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

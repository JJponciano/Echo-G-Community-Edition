# ECHO-G: Evidence-Centered Hybrid Ontologization

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: EGRL v1.0](https://img.shields.io/badge/license-EGRL%20v1.0-lightgrey)](LICENSE)
[![Ollama](https://img.shields.io/badge/inference-Ollama-orange)](https://ollama.com)
[![DOI](https://zenodo.org/badge/1242345747.svg)](https://doi.org/10.5281/zenodo.20269825)

> **ECHO-G: Evidence-Centered Hybrid Ontologization with Semantic Idempotence Evaluation using Local Mixture-of-Experts Language Models**  
> Claire Ponciano and Jean-Jacques Ponciano  
> i3mainz – Institute for Spatial Information and Surveying Technology, Hochschule Mainz  
> *Submitted to WEBIST 2026 (SCITEPRESS)*

---

## Overview

Large language models have made the automatic construction of knowledge graphs from unstructured documents practical, yet the resulting graphs are hard to audit: extracted statements rarely carry traceable evidence, and no standard criterion exists to quantify whether the graph faithfully represents the source.

**ECHO-G** (*Evidence-Centered Hybrid Ontologization – Generic*) addresses both gaps simultaneously:

1. **Auditability** – every assertion is materialised as a named `echo:ExtractedFact` node carrying source span, confidence score, certainty, polarity, validation status, and PROV-O-aligned provenance. Vocabulary outside the user-supplied T-Box is logged as an `echog:OntologyAdditionProposal` rather than silently fabricated.

2. **Quantitative evaluation** – we introduce *semantic idempotence* as an intrinsic faithfulness criterion: the document, after the round trip **PDF → Markdown → RDF/Turtle → Markdown′**, should preserve every source fact and invent none. This is operationalised through LLM-based atomic fact extraction with embedding-based fuzzy matching, yielding recall, precision, and hallucination-rate metrics without requiring a gold-standard ontology.

The entire pipeline — including evaluation — runs on a single workstation through [Ollama](https://ollama.com) using the Qwen3 Mixture-of-Experts open-weights model.

---

## Pipeline

```
  ┌──────────────┐     pdf_to_markdown      ┌────────────────────┐
  │   PDF (src)  │ ───────────────────────► │  Markdown (source) │
  └──────────────┘                          └────────┬───────────┘
                                                     │ pdf_to_ontology
                                                     │ (chunked uplift)
                                                     ▼
  ┌────────────────────┐   ontology_to_markdown  ┌──────────────────────┐
  │ Markdown′ (recon.) │ ◄─────────────────────  │  RDF/Turtle (ECHO-G) │
  └────────┬───────────┘                         └──────────────────────┘
           │
           │ compare_semantics  (cosine similarity, τ = 0.78)
           ▼
  recall · precision · hallucination-rate
```

| Stage | Script | Description |
|-------|--------|-------------|
| PDF → Markdown | `pdf_to_markdown.py` | Extracts text while preserving headings, paragraphs, and page breaks via `pdfplumber`. |
| Markdown → RDF/Turtle | `pdf_to_ontology.py` | Chunked uplift: paragraph-aware splitting, per-chunk IRI prefixes, entity coalescing, exponential-backoff retry. Every assertion is reified as `echo:ExtractedFact`. |
| RDF/Turtle → Markdown′ | `ontology_to_markdown.py` | Traverses the reified graph and reconstructs a natural-language narrative (downlift). |
| Evaluation | `compare_semantics.py` | Extracts atomic facts from both documents via LLM, embeds each with `nomic-embed-text`, and computes fuzzy-match metrics. |
| End-to-end harness | `evaluate_pipeline.py` | Runs the full round trip for each PDF, writes per-document JSON reports and an aggregate `summary.json`. |

---

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU VRAM | **24 GB** (to load Qwen3 MoE in Q4\_K\_M quantisation) |
| RAM | 32 GB recommended |
| OS | Windows 10/11, Linux, macOS |
| Python | 3.12+ |
| Disk | ~20 GB free (model weights + run artefacts) |

The pipeline runs entirely on local hardware; no cloud API key is needed.

---

## Installation

### 1 — Install Ollama

Download and install [Ollama](https://ollama.com/download) for your platform (Windows / Linux / macOS). Once installed, verify it is running:

```bash
ollama list
```

Then pull the two models used by the pipeline:

```bash
# Qwen3 MoE — uplift and atomic-fact extraction
# (Qwen3 30B total parameters, ~3.6B active per token; ~16 GB in Q4_K_M)
ollama pull qwen3.6

# nomic-embed-text — sentence-level semantic similarity
ollama pull nomic-embed-text
```

### 2 — Set up a Python 3.12 environment

**Option A — conda / miniconda (recommended)**

If you do not have conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

```bash
conda create -n echo-g python=3.12 -y
conda activate echo-g
```

**Option B — standard venv**

```bash
python3.12 -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3 — Install Python dependencies

```bash
# From the repo root (installs ollama, pdfplumber, rdflib)
pip install -e .
```

Or without cloning:

```bash
pip install ollama pdfplumber rdflib
```

### 4 — Verify the setup

```bash
python -c "import ollama, pdfplumber, rdflib; print('OK')"
ollama run qwen3.6 "Reply with just the word READY." --nowordwrap
```

---

## Quick Start

### Single document

```bash
# Step 1 – PDF to Markdown
python pdf_to_markdown.py my_document.pdf --out my_document.md

# Step 2 – Uplift: Markdown to RDF/Turtle
python pdf_to_ontology.py my_document.pdf \
    --tbox ontology/echo-core.ttl \
    --out  my_document.ttl

# Step 3 – Downlift: RDF/Turtle to Markdown
python ontology_to_markdown.py my_document.ttl --out my_document_reconstructed.md

# Step 4 – Evaluate semantic idempotence
python compare_semantics.py my_document.md my_document_reconstructed.md
```

### End-to-end evaluation over a folder

```bash
python evaluate_pipeline.py \
    --input-dir eval/sources/hantavirus/ \
    --tbox      ontology/echo-core.ttl \
    --out       eval/runs/my_run
```

Results are written to `eval/runs/my_run/summary.json`.  
Resume after interruption by re-running the same command (completed documents are skipped automatically).  
Use `--force` to reprocess everything.

---

## Reproducing the Multilingual Hantavirus Benchmark

The benchmark corpus ships under `eval/sources/hantavirus/`:

| File | Language | Pages |
|------|----------|------:|
| `Hantavirus.pdf` | English | 15 |
| `Hantaviren.pdf` | German | 22 |
| `Hantavirus_fr.pdf` | French | 7 |

The three PDFs are independent Wikipedia treatments of *Orthohantavirus* (CC BY-SA); they are not translations of one another. To reproduce the published results:

```bash
python evaluate_pipeline.py \
    --input-dir eval/sources/hantavirus/ \
    --tbox      ontology/echo-core.ttl \
    --out       eval/runs/hantavirus
```

The default model (`qwen3.6`) and threshold (τ = 0.78) match the published run exactly; no extra flags are needed.  
Expected wall-clock time: 30–55 minutes per article on a workstation with ≥ 24 GB GPU VRAM.

---

## Results

Results on the multilingual *Hantavirus* corpus (threshold τ = 0.78, `qwen3.6` / Qwen3 30B-A3B Q4\_K\_M):

| Article | Lang | Pages | Facts orig. | Facts rec. | Missing | Halluc. | Recall | Prec. | Halluc. rate |
|---------|------|------:|------------:|-----------:|--------:|--------:|-------:|------:|-------------:|
| *Hantavirus* | en | 15 | 661 | 650 | 15 | 4 | **0.977** | 0.994 | **0.006** |
| *Hantaviren* | de | 22 | 736 | 745 | 50 | 72 | 0.932 | 0.903 | 0.097 |
| *Hantavirus* | fr | 7 | 173 | 187 | 10 | 9 | 0.942 | 0.952 | 0.048 |
| **Macro-avg** | | | | | | | **0.951** | **0.950** | **0.050** |

The dominant failure mode is **cross-lingual paraphrasing, not fabrication**: when the source is German, the downlift sometimes restates facts in English, causing the embedding-based matcher to fall below τ. The extracted graph is not wrong; the round-trip evaluation penalises the surface-form mismatch. See the paper for a detailed qualitative analysis and proposed mitigations.

---

## T-Box (Core Ontology)

All experiments use the generic core T-Box at `ontology/echo-core.ttl`, which contains **69 classes** and **83 properties** (55 object properties, 28 datatype properties). Top-level abstractions include `Document`, `Section`, `Person`, `Organization`, `TimePoint`, `Finding`, `Claim`, `Evidence`, and `ExtractedFact`, covering any document type without committing to a specific domain.

Concept proposals raised by the LLM during extraction are recorded as `echog:OntologyAdditionProposal` and indicate genuine vocabulary gaps with respect to this deliberately under-specified T-Box.

---

## Repository Structure

```
Echo-G-Community-Edition/
├── echo_g_core.py            # Core library (reified extraction, TBox loading, provenance)
├── pdf_to_markdown.py        # Stage 1: PDF → Markdown
├── pdf_to_ontology.py        # Stage 2: Markdown → RDF/Turtle  (chunked uplift)
├── ontology_to_markdown.py   # Stage 3: RDF/Turtle → Markdown′ (downlift)
├── compare_semantics.py      # Stage 4: atomic-fact extraction + fuzzy matching
├── evaluate_pipeline.py      # End-to-end evaluation harness
├── pyproject.toml            # Python package metadata and dependencies
├── ontology/
│   └── echo-core.ttl         # Generic core T-Box (69 classes, 83 properties)
└── eval/
    └── sources/
        └── hantavirus/
            ├── Hantavirus.pdf     # English Wikipedia article (CC BY-SA)
            ├── Hantaviren.pdf     # German Wikipedia article  (CC BY-SA)
            └── Hantavirus_fr.pdf  # French Wikipedia article  (CC BY-SA)
```

---

## Citation

This work builds on:

```bibtex
@inproceedings{ponciano2025ontogrounded,
  author    = {Ponciano, Claire and Schaffert, Markus and Ponciano, Jean-Jacques},
  title     = {{Ontology-Grounded Language Modeling: Enhancing GPT-Based
                Philosophical Text Generation with Structured Knowledge}},
  booktitle = {Proceedings of the 21st International Conference on Web
               Information Systems and Technologies (WEBIST)},
  pages     = {459--467},
  year      = {2025},
  publisher = {SciTePress},
  doi       = {10.5220/0013864400003985}
}

@inproceedings{ponciano2024odkar,
  author    = {Ponciano, Claire and Schaffert, Markus and Ponciano, Jean-Jacques},
  title     = {{ODKAR: Ontology-Based Dynamic Knowledge Acquisition and
                Automated Reasoning Using NLP, OWL, and SWRL}},
  booktitle = {Proceedings of the 20th International Conference on Web
               Information Systems and Technologies (WEBIST)},
  pages     = {457--465},
  year      = {2024},
  publisher = {SciTePress},
  doi       = {10.5220/0013071500003825}
}
```

---

## License

Released under the **Echo-G Community Research License (EGRL) v1.0** — see [LICENSE](LICENSE).  
Free for academic research, education, and scientific publications.  
Commercial use requires prior written permission.

The Hantavirus benchmark PDFs are sourced from Wikipedia under the **CC BY-SA** licence.

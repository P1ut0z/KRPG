# KRPG

KRPG (Knowledge-Retrieval Prompted Generator) is a prototype framework for antimicrobial peptide (AMP) design. It combines a lightweight AMP knowledge graph, retrieval-augmented prompt construction, a prompt-conditioned Transformer sequence generator, and rule-based computational validation.

The current repository is suitable for local CPU verification, dataset preparation, knowledge-graph rebuilding, and server-side training or generation experiments. Generated peptide candidates are computational hypotheses only and require expert review and experimental validation before any downstream use.

## What Is Included

- Knowledge graph and retrieval modules for AMP design rules, peptide records, activity classes, assays, structures, and source evidence.
- Prompt parsing and prompt construction for turning design goals into structured generation context.
- Amino-acid tokenizer and a compact GPT-style causal Transformer generator.
- Prompt-conditioning support through a small `PromptEncoder`.
- Rule-based AMP, toxicity, stability, similarity, physicochemical, and composite scoring utilities.
- Data build scripts for curated AMP positives and weak background examples.
- Local and server-side scripts for end-to-end verification, training, generation, evaluation, and closed-loop optimization.

## Project Layout

```text
KRPG/
  config/
    local_config.yaml          # small CPU-oriented settings
    server_config.yaml         # larger GPU/server-oriented settings
  data/
    README_real_dataset.md     # dataset notes and rebuild commands
    amp_dataset_clean.csv      # main curated AMP training table
    amp_sequences.json         # generation-ready AMP sequences
    knowledge_base/
      README.md                # mature knowledge graph notes
      kg_summary.json          # summary of generated KG statistics
    processed/                 # curated positives and weak background files
    raw/                       # APD6, DRAMP, DBAASP source exports
  krpg/
    knowledge/                 # KnowledgeGraph, RAGRetriever, PromptBuilder, PromptParser
    generation/                # AminoAcidTokenizer, KRPGGenerator, SequenceGenerator
    validation/                # AMP/toxicity/stability/similarity/scoring/feedback
  scripts/
    build_real_dataset.py      # rebuild curated AMP and weak background data
    build_kg.py                # rebuild knowledge graph JSONL files
    run_local.py               # CPU-friendly local end-to-end verification
    run_server.py              # train/generate/evaluate/full pipeline modes
  tests/
    test_modules.py            # module-level pytest coverage
```

Large generated files are intentionally ignored by Git, including `data/knowledge_base/entities.jsonl`, `data/knowledge_base/relations.jsonl`, the UniProt background FASTA archive, Python caches, and model checkpoint files.

## Setup

Use Python 3.10+ or a recent Conda environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

The required runtime dependencies currently listed in `requirements.txt` are:

```text
torch>=1.13.0
numpy>=1.21.0
```

## Quick Start

Run the local CPU verification pipeline:

```bash
python scripts/run_local.py
```

This script exercises:

- knowledge graph construction and retrieval
- structured prompt building
- tokenizer encode/decode
- small prompt-conditioned generator training
- candidate sequence generation
- AMP, toxicity, stability, similarity, physicochemical, and composite scoring
- feedback-based design-spec updates

The local summary is written to:

```text
outputs/local_verification_summary.json
```

## Server-Side Usage

`scripts/run_server.py` provides larger training and generation workflows.

Train a model:

```bash
python scripts/run_server.py --mode train --epochs 50 --batch_size 32
```

Generate candidate AMP sequences:

```bash
python scripts/run_server.py --mode generate --checkpoint outputs/checkpoints/krpg_model_epoch50.pt
```

Evaluate candidate sequences or the default dataset:

```bash
python scripts/run_server.py --mode evaluate
```

Run a single full train-generate-evaluate-feedback pipeline:

```bash
python scripts/run_server.py --mode full
```

Run the closed-loop pipeline alias:

```bash
python scripts/run_server.py --mode pipeline
```

The script automatically uses CUDA when available and falls back to CPU otherwise.

## Rebuild Data

The main generation dataset uses curated positive AMP sequences only. Weak UniProt background fragments are exported separately and should not be treated as experimentally confirmed non-AMPs.

After the raw source files are present in `data/raw`, rebuild the processed datasets with:

```bash
python scripts/build_real_dataset.py --max-generation-len 50 --background-size 20000
```

Current dataset summary:

- generation-ready curated positives: `20227`
- full curated positive library: `21521`
- weak background rows: `20000`
- weak binary rows: `40227`
- retained amino-acid alphabet: standard 20 amino acids
- generation length range: 5 to 50 residues

See `data/README_real_dataset.md` and `data/processed/dataset_summary.json` for source details.

## Rebuild Knowledge Graph

Rebuild the mature knowledge graph from the curated AMP dataset:

```bash
python scripts/build_kg.py --amp-csv data/amp_dataset_clean.csv --out data/knowledge_base
```

This produces:

```text
data/knowledge_base/entities.jsonl
data/knowledge_base/relations.jsonl
data/knowledge_base/kg_summary.json
```

The runtime retrieval layer prefers the mature JSONL graph when `entities.jsonl` and `relations.jsonl` exist locally. These JSONL files can be large, so they are ignored by Git and should be rebuilt when needed.

## Test

Run module tests with:

```bash
python -m pytest tests/test_modules.py -v
```

If `torch` or `numpy` crashes during import, fix the local Python/Conda numerical stack first. That failure happens before project tests run.

## Minimal API Example

```python
from krpg.knowledge import KnowledgeGraph, RAGRetriever, PromptBuilder
from krpg.generation import AminoAcidTokenizer, KRPGGenerator
from krpg.generation.generator import SequenceGenerator
from krpg.validation import RuleBasedAMPPredictor, ToxicityPredictor, StabilityPredictor

kg = KnowledgeGraph()
kg.build_default_amp_knowledge_graph()

retriever = RAGRetriever(kg)
context = retriever.retrieve_by_target(
    "Broad-Spectrum AMP, Gram-Negative, Low Toxicity",
    constraints={"length": "12-20", "charge": "+4 to +8"},
)

builder = PromptBuilder()
design_spec = {
    "target": "Broad-Spectrum AMP",
    "activity": "Gram-Negative",
    "constraint": "Low Toxicity",
    "preference": "Length 12-20, Positive Charge, Amphipathic",
    "knowledge": context,
}
prompt = builder.build_structured_prompt(design_spec)

tokenizer = AminoAcidTokenizer()
model = KRPGGenerator(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=128,
    max_seq_len=64,
    pad_token_id=tokenizer.pad_token_id,
    use_prompt_conditioning=True,
)
generator = SequenceGenerator(model, tokenizer, device="cpu")

amp_predictor = RuleBasedAMPPredictor()
toxicity_predictor = ToxicityPredictor()
stability_predictor = StabilityPredictor()
```

## Version Control Notes

This repository is connected to:

```text
https://github.com/P1ut0z/KRPG.git
```

Before committing, check:

```bash
git status
git diff
```

Keep generated caches, checkpoints, local settings, and very large regenerated data out of Git. The current `.gitignore` already covers the common local artifacts for this project.

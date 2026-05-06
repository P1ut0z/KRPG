# KRPG Real Dataset

This directory now separates the KRPG generation dataset from weak binary
classification background data.

## Main AMP Training Data

- `data/amp_dataset_clean.csv`
- `data/amp_sequences.json`
- `data/processed/amp_generation_train.csv`

These files contain curated AMP positive sequences only. They are deduplicated,
restricted to the 20 standard amino acids, and length-filtered to 5-50 residues
so they match the current generation model configuration.

## Full Curated AMP Library

- `data/processed/amp_curated_positive.csv`

This keeps all deduplicated standard AMP sequences up to 100 residues from the
downloaded AMP sources. Rows longer than 50 residues are retained here for
knowledge retrieval or future model settings, but are excluded from the default
generation training file.

## Weak Background Data

- `data/processed/protein_background_filtered.csv`
- `data/processed/amp_binary_weak.csv`

The background rows are deterministic, length-matched fragments sampled from
reviewed UniProtKB/Swiss-Prot proteins after excluding obvious antimicrobial
keywords. They are marked `label_confidence=weak_background` and should not be
treated as experimentally confirmed non-AMPs.

## Sources

- APD6 natural and animal AMP FASTA exports from `https://aps.unmc.edu/downloads`
- DRAMP 3.0 general and antimicrobial AMP exports from `https://dramp.cpu-bioinfor.org/downloads/`
- DBAASP REST `/peptides` index from `https://dbaasp.org/peptides`
- UniProtKB REST FASTA stream for reviewed non-AMP background proteins from `https://rest.uniprot.org/uniprotkb/`

The exact source URLs and build statistics are saved in
`data/processed/dataset_summary.json`.

## Rebuild

After raw files are present in `data/raw`, rebuild with:

```bash
python scripts/build_real_dataset.py --max-generation-len 50 --background-size 20000
```

Then rebuild the knowledge graph:

```bash
python scripts/build_kg.py --amp-csv data/amp_dataset_clean.csv --out data/knowledge_base
```

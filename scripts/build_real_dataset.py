"""
Build KRPG real datasets from public AMP and protein resources.

The main project dataset is positive-only and generation-ready:
  data/amp_dataset_clean.csv
  data/amp_sequences.json

Weak UniProt background fragments are exported separately for baseline binary
experiments, but they are not mixed into the AMP generation training set.
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_SEED = 20260506

SOURCE_URLS = {
    "APD6 natural": "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
    "APD6 animal": "https://aps.unmc.edu/assets/sequences/animalAMPs_APD2024a.fasta",
    "DRAMP general txt": "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.txt",
    "DRAMP general fasta": "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/general_amps.fasta",
    "DRAMP antimicrobial txt": "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antimicrobial_amps.txt",
    "DRAMP antimicrobial fasta": "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/Antimicrobial_amps.fasta",
    "DBAASP REST": "https://dbaasp.org/peptides?limit=1000&offset={offset}",
    "UniProt background": (
        "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query="
        "%28reviewed%3Atrue%29%20NOT%20%28keyword%3A%22Antimicrobial%20%5BKW-0929%5D%22%20OR%20"
        "keyword%3A%22Antibiotic%20%5BKW-0046%5D%22%20OR%20protein_name%3Adefensin%20OR%20"
        "protein_name%3Acathelicidin%20OR%20protein_name%3Amagainin%20OR%20protein_name%3Acecropin%29"
    ),
}


def normalize_sequence(sequence: str) -> str:
    return "".join(ch for ch in sequence.upper().strip() if ch.isalpha())


def is_standard_peptide(sequence: str, min_len: int = 5, max_len: int = 100) -> bool:
    return min_len <= len(sequence) <= max_len and all(ch in STD_AA for ch in sequence)


def stable_id(prefix: str, sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("ascii")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def parse_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    header = None
    chunks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and chunks:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header and chunks:
            yield header, "".join(chunks)


def first_token(text: str) -> str:
    return (text or "").split()[0].strip()


def add_record(
    records: Dict[str, Dict],
    sequence: str,
    source_database: str,
    source_file: Path,
    database_id: str = "",
    name: str = "",
    activity: str = "",
    target_organism: str = "",
    source_url: str = "",
    origin: str = "",
    synthesis_type: str = "",
    complexity: str = "",
    evidence_note: str = "",
):
    seq = normalize_sequence(sequence)
    if not is_standard_peptide(seq):
        return

    row = records.setdefault(
        seq,
        {
            "sequence": seq,
            "label": 1,
            "label_confidence": "curated_positive",
            "source_database": source_database,
            "source_count": 0,
            "database_ids": [],
            "names": [],
            "activity": "",
            "target": "",
            "target_organism": "",
            "origin": "",
            "synthesis_type": "",
            "complexity": "",
            "source_files": [],
            "source_urls": [],
            "evidence_notes": [],
        },
    )
    row["source_count"] += 1
    if source_database not in row["source_database"].split(";"):
        row["source_database"] = join_unique(row["source_database"], source_database)
    for key, value in [
        ("database_ids", database_id),
        ("names", name),
        ("source_files", source_file.as_posix()),
        ("source_urls", source_url),
        ("evidence_notes", evidence_note),
    ]:
        if value and value not in row[key]:
            row[key].append(value)
    row["activity"] = join_unique(row["activity"], activity)
    row["target"] = join_unique(row["target"], infer_target_categories(activity, target_organism))
    row["target_organism"] = join_unique(row["target_organism"], target_organism)
    row["origin"] = join_unique(row["origin"], origin)
    row["synthesis_type"] = join_unique(row["synthesis_type"], synthesis_type)
    row["complexity"] = join_unique(row["complexity"], complexity)


def join_unique(existing: str, value: str, sep: str = ";") -> str:
    value = (value or "").strip()
    if not value:
        return existing
    parts = [p.strip() for p in existing.split(sep) if p.strip()]
    if value not in parts:
        parts.append(value)
    return sep.join(parts)


def load_apd6(records: Dict[str, Dict]) -> Counter:
    counts = Counter()
    for file_name, label, url in [
        ("naturalAMPs_APD2024a.fasta", "APD6 natural", SOURCE_URLS["APD6 natural"]),
        ("animalAMPs_APD2024a.fasta", "APD6 animal", SOURCE_URLS["APD6 animal"]),
    ]:
        path = RAW / "apd6" / file_name
        if not path.exists():
            continue
        for header, sequence in parse_fasta(path):
            if header.lower().startswith("your search led"):
                continue
            add_record(
                records,
                sequence=sequence,
                source_database="APD6",
                source_file=path.relative_to(ROOT),
                database_id=first_token(header),
                source_url=url,
                origin=label,
                evidence_note="APD6 FASTA export",
            )
            counts[label] += 1
    return counts


def load_dramp(records: Dict[str, Dict]) -> Counter:
    counts = Counter()
    source_url = SOURCE_URLS["DRAMP general txt"]
    txt_path = RAW / "dramp" / "general_amps.txt"
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                add_record(
                    records,
                    sequence=row.get("Sequence", ""),
                    source_database="DRAMP",
                    source_file=txt_path.relative_to(ROOT),
                    database_id=row.get("DRAMP_ID", ""),
                    name=row.get("Name", ""),
                    activity=row.get("Activity", ""),
                    target_organism=row.get("Target_Organism", ""),
                    source_url=source_url,
                    origin=row.get("Source", ""),
                    synthesis_type=row.get("Stereochemistry", ""),
                    complexity=row.get("Linear/Cyclic/Branched", ""),
                    evidence_note=clean_text(row.get("Reference", ""))[:300],
                )
                counts["DRAMP general txt"] += 1

    for file_name, label, url in [
        ("general_amps.fasta", "DRAMP general fasta", SOURCE_URLS["DRAMP general fasta"]),
        ("Antimicrobial_amps.fasta", "DRAMP antimicrobial fasta", SOURCE_URLS["DRAMP antimicrobial fasta"]),
    ]:
        path = RAW / "dramp" / file_name
        if not path.exists():
            continue
        for header, sequence in parse_fasta(path):
            add_record(
                records,
                sequence=sequence,
                source_database="DRAMP",
                source_file=path.relative_to(ROOT),
                database_id=first_token(header),
                source_url=url,
                activity="Antimicrobial" if "Antimicrobial" in file_name else "",
                evidence_note=label,
            )
            counts[label] += 1
    return counts


def load_dbaasp(records: Dict[str, Dict]) -> Counter:
    counts = Counter()
    for path in sorted((RAW / "dbaasp").glob("peptides_offset_*_limit_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.get("data", []):
            sequences = peptide_sequences_from_dbaasp_item(item)
            for seq, seq_name in sequences:
                add_record(
                    records,
                    sequence=seq,
                    source_database="DBAASP",
                    source_file=path.relative_to(ROOT),
                    database_id=str(item.get("dbaaspId") or item.get("id") or ""),
                    name=seq_name or item.get("name", ""),
                    source_url="https://dbaasp.org/peptides",
                    origin=str(item.get("synthesisType") or ""),
                    synthesis_type=str(item.get("synthesisType") or ""),
                    complexity=str(item.get("complexity") or ""),
                    evidence_note="DBAASP REST /peptides index",
                )
                counts["DBAASP peptide or monomer"] += 1
    return counts


def peptide_sequences_from_dbaasp_item(item: Dict) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    seq = item.get("sequence")
    if seq:
        result.append((seq, item.get("name", "")))
    for monomer in item.get("monomers") or []:
        monomer_seq = monomer.get("sequence")
        if monomer_seq:
            result.append((monomer_seq, monomer.get("name", "") or item.get("name", "")))
    return result


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def infer_target_categories(activity: str, target_organism: str = "") -> str:
    text = f"{activity} {target_organism}".lower()
    categories = []
    patterns = [
        ("Gram-positive bacteria", ["gram+", "gram-positive", "gram positive"]),
        ("Gram-negative bacteria", ["gram-", "gram-negative", "gram negative"]),
        ("Bacteria", ["antibacterial", "bacteri", "staphylococcus", "escherichia", "pseudomonas"]),
        ("Fungi", ["antifungal", "fung", "candida", "aspergillus"]),
        ("Virus", ["antiviral", "virus", "viral", "sars-cov-2", "hiv"]),
        ("Cancer cells", ["anticancer", "tumor", "cancer", "carcinoma", "melanoma"]),
        ("Parasites", ["antiparasitic", "parasite", "plasmodium", "leishmania"]),
        ("Biofilm", ["biofilm"]),
    ]
    for label, needles in patterns:
        if any(needle in text for needle in needles):
            categories.append(label)
    return ";".join(categories)


def finalize_amp_rows(records: Dict[str, Dict], max_generation_len: int) -> List[Dict]:
    rows = []
    for seq, row in records.items():
        out = dict(row)
        out["sequence_length"] = len(seq)
        out["is_generation_ready"] = 5 <= len(seq) <= max_generation_len
        for list_key in ["database_ids", "names", "source_files", "source_urls", "evidence_notes"]:
            out[list_key] = ";".join(out[list_key])
        out["primary_database_id"] = out["database_ids"].split(";")[0] if out["database_ids"] else stable_id("AMP", seq)
        out["primary_name"] = out["names"].split(";")[0] if out["names"] else out["primary_database_id"]
        out["source"] = out["source_database"]
        out["database_id"] = out["primary_database_id"]
        out["name"] = out["primary_name"]
        rows.append(out)
    return sorted(rows, key=lambda r: (r["sequence_length"], r["sequence"]))


def parse_uniprot_fasta_gz(path: Path) -> Iterator[Tuple[str, str, str]]:
    header = None
    chunks: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and chunks:
                    yield parse_uniprot_header(header), header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header and chunks:
            yield parse_uniprot_header(header), header, "".join(chunks)


def parse_uniprot_header(header: str) -> str:
    match = re.match(r"\w+\|([^|]+)\|", header)
    return match.group(1) if match else first_token(header)


def build_background_rows(
    positive_sequences: set,
    length_distribution: List[int],
    max_rows: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    path = RAW / "uniprot" / "uniprot_sprot_reviewed_non_amp_background.fasta.gz"
    if not path.exists() or not length_distribution:
        return []

    target_lengths = list(length_distribution)
    rng.shuffle(target_lengths)
    rows: List[Dict] = []
    seen = set(positive_sequences)
    protein_pool = []

    for accession, header, protein_seq in parse_uniprot_fasta_gz(path):
        seq = normalize_sequence(protein_seq)
        if len(seq) < 80 or not all(ch in STD_AA for ch in seq):
            continue
        protein_pool.append((accession, header, seq))
        if len(protein_pool) >= max_rows * 3:
            break

    if not protein_pool:
        return []

    attempts = 0
    max_attempts = max_rows * 40
    while len(rows) < max_rows and attempts < max_attempts:
        attempts += 1
        accession, header, protein_seq = protein_pool[attempts % len(protein_pool)]
        length = target_lengths[attempts % len(target_lengths)]
        if length >= len(protein_seq):
            continue
        start = rng.randint(0, len(protein_seq) - length)
        fragment = protein_seq[start:start + length]
        if fragment in seen or not is_standard_peptide(fragment, min_len=5, max_len=100):
            continue
        seen.add(fragment)
        rows.append({
            "sequence": fragment,
            "label": 0,
            "label_confidence": "weak_background",
            "source_database": "UniProtKB/Swiss-Prot",
            "source": "UniProtKB/Swiss-Prot",
            "database_id": accession,
            "name": accession,
            "sequence_length": len(fragment),
            "parent_protein_accession": accession,
            "parent_header": header,
            "fragment_start_1based": start + 1,
            "fragment_end_1based": start + length,
            "sampling_strategy": "reviewed_non_amp_swissprot_length_matched_fragment",
            "source_url": SOURCE_URLS["UniProt background"],
            "note": "Weak background only; not treated as experimentally confirmed non-AMP.",
        })
    return rows


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build KRPG real AMP datasets")
    parser.add_argument("--max-generation-len", type=int, default=50)
    parser.add_argument("--background-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    records: Dict[str, Dict] = {}
    source_counts = Counter()
    source_counts.update(load_apd6(records))
    source_counts.update(load_dramp(records))
    source_counts.update(load_dbaasp(records))

    all_positive = finalize_amp_rows(records, max_generation_len=args.max_generation_len)
    generation_ready = [r for r in all_positive if r["is_generation_ready"]]
    positive_sequences = {r["sequence"] for r in all_positive}

    background_rows = build_background_rows(
        positive_sequences=positive_sequences,
        length_distribution=[r["sequence_length"] for r in generation_ready],
        max_rows=min(args.background_size, len(generation_ready)),
        seed=args.seed,
    )

    amp_fields = [
        "sequence", "label", "label_confidence", "sequence_length", "is_generation_ready",
        "source", "source_database", "source_count", "database_id", "name",
        "database_ids", "names", "activity", "target", "target_organism", "origin",
        "synthesis_type", "complexity", "source_files", "source_urls", "evidence_notes",
    ]
    train_fields = [
        "sequence", "label", "label_confidence", "sequence_length",
        "source", "source_database", "source_count", "database_id", "name",
        "activity", "target", "target_organism", "origin", "synthesis_type", "complexity",
        "source_files", "source_urls", "evidence_notes",
    ]

    write_csv(PROCESSED / "amp_curated_positive.csv", all_positive, amp_fields)
    write_csv(PROCESSED / "amp_generation_train.csv", generation_ready, train_fields)
    write_csv(DATA / "amp_dataset_clean.csv", generation_ready, train_fields)
    write_json(DATA / "amp_sequences.json", [{"sequence": r["sequence"], "source": r["source_database"]} for r in generation_ready])
    write_csv(PROCESSED / "protein_background_filtered.csv", background_rows)

    binary_rows = generation_ready + background_rows[:len(generation_ready)]
    write_csv(PROCESSED / "amp_binary_weak.csv", binary_rows)

    length_counter = Counter(r["sequence_length"] for r in generation_ready)
    summary = {
        "build_seed": args.seed,
        "max_generation_len": args.max_generation_len,
        "source_urls": SOURCE_URLS,
        "raw_source_counts": dict(source_counts),
        "curated_positive_unique": len(all_positive),
        "generation_ready_positive_unique": len(generation_ready),
        "weak_background_rows": len(background_rows),
        "binary_weak_rows": len(binary_rows),
        "source_database_counts_generation_ready": dict(Counter(r["source_database"] for r in generation_ready)),
        "length_min_generation_ready": min(length_counter) if length_counter else None,
        "length_max_generation_ready": max(length_counter) if length_counter else None,
        "length_median_generation_ready": percentile([r["sequence_length"] for r in generation_ready], 0.5),
        "outputs": {
            "main_csv": "data/amp_dataset_clean.csv",
            "sequence_json": "data/amp_sequences.json",
            "curated_positive_csv": "data/processed/amp_curated_positive.csv",
            "generation_train_csv": "data/processed/amp_generation_train.csv",
            "protein_background_csv": "data/processed/protein_background_filtered.csv",
            "binary_weak_csv": "data/processed/amp_binary_weak.csv",
        },
        "notes": [
            "Main KRPG generation dataset uses curated AMP positives only.",
            "UniProt fragments are weak background examples and should not be read as experimentally validated non-AMPs.",
            "Only standard 20 amino-acid sequences are retained.",
        ],
    }
    write_json(PROCESSED / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def percentile(values: List[int], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return float(vals[idx])


if __name__ == "__main__":
    main()

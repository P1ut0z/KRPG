import csv
from typing import Dict, List, Optional


def load_amp_csv_records(path: str, max_records: Optional[int] = None) -> List[Dict]:
    """Load a simple AMP CSV with at least a sequence column.

    This loader intentionally keeps the source schema lightweight so the current
    project can build a useful peptide KG from `data/amp_dataset_clean.csv`, and
    later reuse the same normalized record shape for DRAMP/DBAASP loaders.
    """
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = (row.get("sequence") or row.get("Sequence") or "").strip().upper()
            if not sequence:
                continue
            label_raw = row.get("label", row.get("Label"))
            label = None
            if label_raw not in (None, ""):
                try:
                    label = int(float(label_raw))
                except ValueError:
                    label = label_raw
            record = {
                "sequence": sequence,
                "label": label,
                "source": row.get("source") or row.get("source_database") or path,
                "raw": dict(row),
            }
            for key in [
                "database_id",
                "name",
                "source_database",
                "source_url",
                "source_urls",
                "source_files",
                "activity",
                "target",
                "target_organism",
                "label_confidence",
                "sequence_length",
                "is_generation_ready",
                "origin",
                "synthesis_type",
                "complexity",
                "database_ids",
                "names",
                "evidence_notes",
            ]:
                value = row.get(key) or row.get(key.title())
                if value not in (None, ""):
                    record[key] = value
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    return records

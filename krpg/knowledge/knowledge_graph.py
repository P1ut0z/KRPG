import csv
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from krpg.knowledge.schema import EntityRecord, Evidence, RelationRecord


class KnowledgeGraph:
    def __init__(self, data_dir: Optional[str] = None):
        self.entities: Dict[str, dict] = {}
        self.relations: List[Tuple[str, str, str, dict]] = []
        self._relation_keys = set()
        self.enrichment_summary: Dict[str, int] = {}
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base")

    def add_entity(self, entity_id: str, entity_type: str, properties: dict = None,
                   name: Optional[str] = None, evidence: Optional[List[dict]] = None):
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            entity["properties"].update(properties or {})
            if name and entity.get("name") == entity_id:
                entity["name"] = name
            existing_evidence = entity.setdefault("evidence", [])
            for item in evidence or []:
                if item not in existing_evidence:
                    existing_evidence.append(item)
            return
        self.entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "name": name or entity_id,
            "properties": dict(properties or {}),
            "evidence": list(evidence or []),
        }

    def add_relation(self, head: str, relation: str, tail: str, properties: dict = None):
        if head not in self.entities:
            raise ValueError(f"Entity '{head}' not found in graph")
        if tail not in self.entities:
            raise ValueError(f"Entity '{tail}' not found in graph")
        props = properties or {}
        relation_key = (head, relation, tail, json.dumps(props, ensure_ascii=False, sort_keys=True, default=str))
        if relation_key in self._relation_keys:
            return
        self._relation_keys.add(relation_key)
        self.relations.append((head, relation, tail, props))

    def add_record_entity(self, entity: EntityRecord):
        self.add_entity(
            entity.id,
            entity.type,
            properties=entity.properties,
            name=entity.name,
            evidence=entity.evidence,
        )

    def add_record_relation(self, relation: RelationRecord):
        h, r, t, props = relation.to_tuple()
        self.add_relation(h, r, t, props)

    @staticmethod
    def _normalize_sequence(sequence: str) -> str:
        return "".join(aa for aa in sequence.upper().strip() if aa.isalpha())

    @staticmethod
    def _is_standard_peptide(sequence: str) -> bool:
        return bool(sequence) and all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _is_missing_text(value: Optional[str]) -> bool:
        text = KnowledgeGraph._clean_text(value).lower()
        if not text:
            return True
        missing_markers = [
            "not found",
            "no entry found",
            "no information",
            "no data found",
            "not included yet",
            "not metioned clearly",
            "not mentioned clearly",
        ]
        return any(marker in text for marker in missing_markers)

    @staticmethod
    def _stable_hash(value: str, n: int = 12) -> str:
        return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]

    @staticmethod
    def _safe_id(prefix: str, value: str, max_value_len: int = 80) -> str:
        clean = KnowledgeGraph._clean_text(value)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", clean.lower()).strip("_")
        if not slug:
            slug = KnowledgeGraph._stable_hash(clean or prefix)
        if len(slug) > max_value_len:
            slug = f"{slug[:max_value_len]}_{KnowledgeGraph._stable_hash(clean, 8)}"
        return f"{prefix}:{slug}"

    @staticmethod
    def _split_values(value: Optional[str]) -> List[str]:
        text = KnowledgeGraph._clean_text(value)
        if KnowledgeGraph._is_missing_text(text):
            return []
        parts = re.split(r";|\|\||##", text)
        return [p.strip() for p in parts if p.strip() and not KnowledgeGraph._is_missing_text(p)]

    @staticmethod
    def _split_activity_terms(value: Optional[str]) -> List[str]:
        text = KnowledgeGraph._clean_text(value)
        if KnowledgeGraph._is_missing_text(text):
            return []
        parts = re.split(r";|,", text)
        return [p.strip() for p in parts if p.strip() and not KnowledgeGraph._is_missing_text(p)]

    @staticmethod
    def _infer_database_from_id(database_id: str) -> str:
        dbid = database_id.upper()
        if dbid.startswith("DRAMP"):
            return "DRAMP"
        if dbid.startswith("DBAASP"):
            return "DBAASP"
        if dbid.startswith("AP"):
            return "APD6"
        if re.match(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$", database_id):
            return "UniProtKB/Swiss-Prot"
        return "database"

    @staticmethod
    def _activity_class_label(term: str) -> str:
        lower = term.lower()
        mapping = [
            ("anti_gram_positive", "Anti-Gram+"),
            ("anti_gram_negative", "Anti-Gram-"),
            ("antibacterial", "Antibacterial"),
            ("antifungal", "Antifungal"),
            ("antiviral", "Antiviral"),
            ("anticancer", "Anticancer"),
            ("antiparasitic", "Antiparasitic"),
            ("antibiofilm", "Antibiofilm"),
            ("insecticidal", "Insecticidal"),
            ("antimicrobial", "Antimicrobial"),
        ]
        for canonical, needle in mapping:
            if needle.lower() in lower:
                return canonical
        return re.sub(r"[^a-z0-9]+", "_", lower).strip("_") or "activity"

    @staticmethod
    def _target_classes_from_text(*parts: str) -> List[str]:
        text = " ".join(KnowledgeGraph._clean_text(p).lower() for p in parts)
        patterns = [
            ("Gram-positive bacteria", ["gram+", "gram-positive", "gram positive"]),
            ("Gram-negative bacteria", ["gram-", "gram-negative", "gram negative"]),
            ("Bacteria", ["antibacterial", "bacteri", "staphylococcus", "escherichia", "pseudomonas", "bacillus"]),
            ("Fungi", ["antifungal", "fung", "candida", "aspergillus"]),
            ("Virus", ["antiviral", "virus", "viral", "sars-cov-2", "hiv"]),
            ("Cancer cells", ["anticancer", "tumor", "tumour", "cancer", "carcinoma", "melanoma"]),
            ("Parasites", ["antiparasitic", "parasite", "plasmodium", "leishmania"]),
            ("Biofilm", ["biofilm"]),
        ]
        labels = []
        for label, needles in patterns:
            if any(needle in text for needle in needles):
                labels.append(label)
        return labels

    @staticmethod
    def _extract_measurement(text: str) -> Optional[dict]:
        raw = KnowledgeGraph._clean_text(text)
        pattern = re.compile(
            r"\b(MIC90|MIC50|MIC|MBC|IC50|IC|EC50|LC50|HC50|MHC|LD50)\b\s*"
            r"(<=|>=|≤|≥|<|>|=|鈮?|~)?\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*"
            r"([a-zA-Zµμ渭/%\.\-]+(?:/[a-zA-Z%]+)?)?",
            re.IGNORECASE,
        )
        match = pattern.search(raw)
        if not match:
            return None
        operator = match.group(2) or "="
        if operator == "鈮?":
            operator = "<="
        unit = (match.group(4) or "").replace("μ", "u").replace("µ", "u").replace("渭", "u")
        return {
            "measure": match.group(1).upper(),
            "operator": operator,
            "value": float(match.group(3)),
            "unit": unit,
            "raw": raw,
        }

    @staticmethod
    def _extract_target_entries(target_text: str, activity: str = "", max_entries: int = 30) -> List[dict]:
        text = KnowledgeGraph._clean_text(target_text)
        if KnowledgeGraph._is_missing_text(text) or "no mics found" in text.lower():
            return []
        normalized = text.replace("##", ";").replace("\n", " ")
        segments = [s.strip() for s in re.split(r";|\.\s+", normalized) if s.strip()]
        entries = []
        for segment in segments:
            category_hint = ""
            body = segment
            if ":" in segment:
                category_hint, body = segment.split(":", 1)
            raw_items = re.split(r"\),\s*|,\s*(?=[A-Z][a-z]+)", body)
            for raw_item in raw_items:
                item = raw_item.strip(" .;")
                if not item or KnowledgeGraph._is_missing_text(item):
                    continue
                measurement = KnowledgeGraph._extract_measurement(item)
                name = re.sub(r"\([^)]*(?:MIC|MBC|IC|EC|LC|HC|MHC|LD)[^)]*\)", "", item, flags=re.IGNORECASE)
                name = re.sub(r"\[[^\]]+\]", "", name)
                name = KnowledgeGraph._clean_text(name.strip(" ,.;:"))
                binomial = re.search(r"\b([A-Z][a-z]+(?:\s+[a-z][a-z\-]+){1,2}(?:\s+[A-Z0-9][A-Za-z0-9\-]*)?)", name)
                if binomial:
                    name = binomial.group(1)
                if len(name) < 3 or name.lower().startswith(("ref", "human pathogens")):
                    continue
                if len(name) > 120:
                    name = f"{name[:100]}..."
                entries.append({
                    "target": name,
                    "category_hint": category_hint,
                    "target_classes": KnowledgeGraph._target_classes_from_text(category_hint, name, activity),
                    "measurement": measurement,
                    "raw": item,
                })
                if len(entries) >= max_entries:
                    return entries
        return entries

    @staticmethod
    def _compute_physicochemical_properties(sequence: str) -> dict:
        """Compute basic peptide properties without importing the torch stack."""
        seq = sequence.upper()
        length = len(seq)
        if length == 0:
            return {
                "length": 0,
                "molecular_weight": 0.0,
                "net_charge": 0,
                "hydrophobicity": 0.0,
                "isoelectric_point": 0.0,
                "positive_residues": 0,
                "negative_residues": 0,
                "hydrophobic_ratio": 0.0,
                "aromatic_ratio": 0.0,
            }

        molecular_weight_table = {
            "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
            "G": 75.07, "H": 155.16, "I": 131.18, "K": 146.19, "L": 131.18,
            "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
            "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
        }
        hydrophobicity_table = {
            "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
            "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
            "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
            "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
        }
        mw = sum(molecular_weight_table.get(aa, 0) for aa in seq) - 18.015 * (length - 1)
        n_pos = sum(1 for aa in seq if aa in "KRH")
        n_neg = sum(1 for aa in seq if aa in "DE")
        avg_hydro = sum(hydrophobicity_table.get(aa, 0) for aa in seq) / length
        hydrophobic_ratio = sum(1 for aa in seq if aa in "AILMFWV") / length
        aromatic_ratio = sum(1 for aa in seq if aa in "FWY") / length

        return {
            "length": length,
            "molecular_weight": round(mw, 2),
            "net_charge": n_pos - n_neg,
            "hydrophobicity": round(avg_hydro, 3),
            "isoelectric_point": round(KnowledgeGraph._compute_isoelectric_point(seq), 2),
            "positive_residues": n_pos,
            "negative_residues": n_neg,
            "hydrophobic_ratio": round(hydrophobic_ratio, 3),
            "aromatic_ratio": round(aromatic_ratio, 3),
        }

    @staticmethod
    def _compute_isoelectric_point(sequence: str) -> float:
        pka = {
            "N_term": 9.69,
            "C_term": 2.34,
            "K": 10.53,
            "R": 12.48,
            "H": 6.00,
            "D": 3.65,
            "E": 4.25,
            "C": 8.18,
            "Y": 10.07,
        }

        def charge_at_ph(ph: float) -> float:
            charge = 1.0 / (1.0 + 10 ** (ph - pka["N_term"]))
            charge -= 1.0 / (1.0 + 10 ** (pka["C_term"] - ph))
            for aa in sequence:
                if aa in "KRH":
                    charge += 1.0 / (1.0 + 10 ** (ph - pka[aa]))
                elif aa in "DE":
                    charge -= 1.0 / (1.0 + 10 ** (pka[aa] - ph))
                elif aa == "C":
                    charge -= 1.0 / (1.0 + 10 ** (pka["C"] - ph))
                elif aa == "Y":
                    charge -= 1.0 / (1.0 + 10 ** (pka["Y"] - ph))
            return charge

        low, high = 0.0, 14.0
        for _ in range(50):
            mid = (low + high) / 2
            if charge_at_ph(mid) > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def add_peptide(self, sequence: str, peptide_id: Optional[str] = None,
                    properties: Optional[dict] = None, source: str = "unknown",
                    evidence: Optional[List[dict]] = None) -> Optional[str]:
        seq = self._normalize_sequence(sequence)
        if not self._is_standard_peptide(seq):
            return None
        pid = peptide_id or f"peptide:{seq}"
        props = dict(properties or {})
        props.setdefault("sequence", seq)
        props.setdefault("length", len(seq))
        props.setdefault("source", source)
        self.add_entity(pid, "peptide", props, name=seq, evidence=evidence)
        return pid

    def add_reference(self, reference_id: str, properties: Optional[dict] = None) -> str:
        rid = f"reference:{reference_id}"
        self.add_entity(rid, "literature_reference", properties or {}, name=reference_id)
        return rid

    def add_design_rule(self, rule_id: str, description: str,
                        properties: Optional[dict] = None,
                        evidence: Optional[List[dict]] = None) -> str:
        rid = f"rule:{rule_id}"
        props = dict(properties or {})
        props["description"] = description
        self.add_entity(rid, "design_rule", props, name=rule_id, evidence=evidence)
        return rid

    def add_database_record(self, database_id: str, database: Optional[str] = None,
                            properties: Optional[dict] = None,
                            evidence: Optional[List[dict]] = None) -> str:
        dbid = self._clean_text(database_id)
        database = database or self._infer_database_from_id(dbid)
        rid = self._safe_id("database_record", f"{database}:{dbid}")
        props = dict(properties or {})
        props.setdefault("database", database)
        props.setdefault("database_id", dbid)
        self.add_entity(rid, "database_record", props, name=f"{database}:{dbid}", evidence=evidence)
        return rid

    def add_activity_class(self, activity_term: str, evidence: Optional[List[dict]] = None) -> Optional[str]:
        term = self._clean_text(activity_term)
        if not term or self._is_missing_text(term):
            return None
        canonical = self._activity_class_label(term)
        aid = f"activity:{canonical}"
        self.add_entity(aid, "activity_class", {"label": term, "canonical": canonical}, name=term)
        return aid

    def add_target_class(self, target_class: str, evidence: Optional[List[dict]] = None) -> Optional[str]:
        label = self._clean_text(target_class)
        if not label or self._is_missing_text(label):
            return None
        tid = self._safe_id("target_class", label)
        self.add_entity(tid, "target_class", {"label": label}, name=label)
        return tid

    def add_target_organism(self, target: str, evidence: Optional[List[dict]] = None,
                            properties: Optional[dict] = None) -> Optional[str]:
        label = self._clean_text(target)
        if not label or self._is_missing_text(label):
            return None
        tid = self._safe_id("target", label)
        props = dict(properties or {})
        props.setdefault("name", label)
        self.add_entity(tid, "target_organism", props, name=label)
        return tid

    def add_literature_reference(self, reference_id: str, properties: Optional[dict] = None,
                                 evidence: Optional[List[dict]] = None) -> Optional[str]:
        rid_text = self._clean_text(reference_id)
        if not rid_text or self._is_missing_text(rid_text):
            return None
        if rid_text.isdigit():
            rid = f"reference:pubmed:{rid_text}"
            props = {"pubmed_id": rid_text, "url": f"https://pubmed.ncbi.nlm.nih.gov/{rid_text}/"}
        else:
            rid = f"reference:text:{self._stable_hash(rid_text)}"
            props = {"reference_text": rid_text}
        props.update(properties or {})
        self.add_entity(rid, "literature_reference", props, name=rid_text, evidence=evidence)
        return rid

    def add_structure_feature(self, label: str, properties: Optional[dict] = None,
                              evidence: Optional[List[dict]] = None) -> Optional[str]:
        text = self._clean_text(label)
        if not text or self._is_missing_text(text):
            return None
        sid = self._safe_id("structure", text)
        props = dict(properties or {})
        props.setdefault("label", text)
        self.add_entity(sid, "structure_feature", props, name=text, evidence=evidence)
        return sid

    def add_assay(self, assay_type: str, record_id: str, index: int,
                  properties: Optional[dict] = None,
                  evidence: Optional[List[dict]] = None) -> str:
        aid = self._safe_id(f"assay:{assay_type}", f"{record_id}:{index}", max_value_len=100)
        props = dict(properties or {})
        props.setdefault("assay_type", assay_type)
        props.setdefault("source_record", record_id)
        self.add_entity(aid, f"{assay_type}_assay", props, name=f"{assay_type}:{record_id}:{index}", evidence=evidence)
        return aid

    def build_from_amp_records(self, records: Iterable[dict], source: str = "amp_dataset",
                               include_properties: bool = True) -> int:
        """Add peptide entities from normalized AMP records.

        Records should contain at least `sequence`, and may contain `label`,
        `source`, `target`, `activity`, or assay metadata. This is the bridge
        from flat AMP datasets into the report's KG layer.
        """
        n_added = 0
        phys_filter = self._compute_physicochemical_properties if include_properties else None

        for idx, rec in enumerate(records):
            seq = self._normalize_sequence(str(rec.get("sequence", "")))
            if not self._is_standard_peptide(seq):
                continue

            props = {
                "sequence": seq,
                "length": len(seq),
                "label": rec.get("label"),
                "source": rec.get("source", source),
            }
            for key in [
                "database_id",
                "name",
                "source_database",
                "source_url",
                "label_confidence",
                "sequence_length",
                "origin",
                "synthesis_type",
                "complexity",
                "activity",
                "target",
                "target_organism",
            ]:
                value = rec.get(key)
                if value in (None, ""):
                    continue
                if key not in {"sequence_length"}:
                    cleaned_parts = self._split_values(value)
                    if cleaned_parts:
                        value = ";".join(cleaned_parts)
                    elif self._is_missing_text(value):
                        continue
                props[key] = value
            if phys_filter:
                props.update(phys_filter(seq))

            evidence = [Evidence(
                source=str(rec.get("source", source)),
                reference_id=rec.get("database_id"),
                url=rec.get("source_url"),
                confidence=0.9 if rec.get("label_confidence") == "curated_positive" else 0.8,
            ).to_dict()]
            pid = self.add_peptide(seq, properties=props, source=source, evidence=evidence)
            if not pid:
                continue

            if rec.get("label") == 1:
                amp_label_id = "activity_label:amp_positive"
                if amp_label_id not in self.entities:
                    self.add_entity(
                        amp_label_id,
                        "activity_label",
                        {"description": "Dataset positive AMP label"},
                        name="AMP positive",
                    )
                self.add_relation(pid, "has_activity", amp_label_id, {
                    "source": rec.get("source", source),
                    "evidence": evidence,
                    "confidence": 0.8,
                })

            database_ids = self._split_values(rec.get("database_id") or rec.get("database_ids"))
            source_urls = self._split_values(rec.get("source_url") or rec.get("source_urls"))
            for db_idx, database_id in enumerate(database_ids):
                database = self._infer_database_from_id(database_id)
                db_evidence = [Evidence(
                    source=database,
                    reference_id=database_id,
                    url=source_urls[db_idx] if db_idx < len(source_urls) else None,
                    confidence=0.95,
                ).to_dict()]
                db_record_id = self.add_database_record(
                    database_id,
                    database=database,
                    properties={
                        "sequence": seq,
                        "name": rec.get("name"),
                        "source_url": source_urls[db_idx] if db_idx < len(source_urls) else None,
                    },
                    evidence=db_evidence,
                )
                self.add_relation(pid, "mapped_to", db_record_id, {"evidence": db_evidence, "confidence": 0.95})
                self.add_relation(pid, "derived_from", db_record_id, {"source_database": database})

            for activity_term in self._split_activity_terms(rec.get("activity")):
                activity_id = self.add_activity_class(activity_term, evidence=evidence)
                if activity_id:
                    self.add_relation(pid, "has_activity_class", activity_id, {
                        "source": rec.get("source", source),
                        "evidence": evidence,
                        "confidence": 0.85,
                    })

            for target_class in self._target_classes_from_text(rec.get("activity", ""), rec.get("target", ""), rec.get("target_organism", "")):
                target_class_id = self.add_target_class(target_class, evidence=evidence)
                if target_class_id:
                    self.add_relation(pid, "active_against", target_class_id, {
                        "activity": rec.get("activity"),
                        "target_level": "class",
                        "evidence": evidence,
                        "confidence": 0.75,
                    })

            targets_raw = rec.get("target") or rec.get("target_organism")
            targets = [
                t.strip()
                for t in str(targets_raw or "").replace("|", ";").split(";")
                if t.strip()
            ]
            for target in targets[:8]:
                target_id = self.add_target_organism(target, evidence=evidence)
                if not target_id:
                    continue
                self.add_relation(pid, "active_against", target_id, {
                    "activity": rec.get("activity"),
                    "target_level": "organism_or_category",
                    "evidence": evidence,
                    "confidence": rec.get("confidence", 0.8),
                })
            n_added += 1
        return n_added

    def enrich_from_dramp_tsv(self, path: str, max_activity_assays_per_record: int = 30,
                              max_toxicity_assays_per_record: int = 8) -> Dict[str, int]:
        """Enrich peptide KG with DRAMP assay, target, structure, and citation data."""
        stats = {
            "dramp_rows_seen": 0,
            "dramp_rows_matched": 0,
            "database_records": 0,
            "activity_assays": 0,
            "toxicity_assays": 0,
            "references": 0,
            "structure_features": 0,
        }
        if not path or not os.path.exists(path):
            return stats

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                stats["dramp_rows_seen"] += 1
                seq = self._normalize_sequence(row.get("Sequence", ""))
                pid = f"peptide:{seq}"
                if not self._is_standard_peptide(seq) or pid not in self.entities:
                    continue
                stats["dramp_rows_matched"] += 1

                dramp_id = self._clean_text(row.get("DRAMP_ID")) or f"DRAMP:{self._stable_hash(seq)}"
                evidence = [Evidence(
                    source="DRAMP",
                    reference_id=dramp_id,
                    url="https://dramp.cpu-bioinfor.org/",
                    confidence=0.95,
                ).to_dict()]
                db_record_id = self.add_database_record(
                    dramp_id,
                    database="DRAMP",
                    properties={
                        "sequence": seq,
                        "name": self._clean_text(row.get("Name")),
                        "family": self._clean_text(row.get("Family")),
                        "gene": self._clean_text(row.get("Gene")),
                        "source_organism": self._clean_text(row.get("Source")),
                        "activity": self._clean_text(row.get("Activity")),
                        "structure": self._clean_text(row.get("Structure")),
                        "pdb_id": self._clean_text(row.get("PDB_ID")),
                    },
                    evidence=evidence,
                )
                stats["database_records"] += 1
                self.add_relation(pid, "mapped_to", db_record_id, {"evidence": evidence, "confidence": 0.95})

                for ref_id in self._split_values(row.get("Pubmed_ID")):
                    for pmid in re.findall(r"\d+", ref_id):
                        reference_id = self.add_literature_reference(pmid, evidence=evidence)
                        if reference_id:
                            stats["references"] += 1
                            self.add_relation(pid, "supported_by", reference_id, {"source": "DRAMP", "database_record": dramp_id})
                            self.add_relation(db_record_id, "supported_by", reference_id, {"source": "DRAMP"})
                if row.get("Reference") and not self._is_missing_text(row.get("Reference")):
                    reference_id = self.add_literature_reference(
                        self._clean_text(row.get("Reference"))[:500],
                        properties={
                            "title": self._clean_text(row.get("Title")),
                            "author": self._clean_text(row.get("Author")),
                        },
                        evidence=evidence,
                    )
                    if reference_id:
                        stats["references"] += 1
                        self.add_relation(pid, "supported_by", reference_id, {"source": "DRAMP", "database_record": dramp_id})

                for activity_term in self._split_activity_terms(row.get("Activity")):
                    activity_id = self.add_activity_class(activity_term, evidence=evidence)
                    if activity_id:
                        self.add_relation(pid, "has_activity_class", activity_id, {
                            "source": "DRAMP",
                            "database_record": dramp_id,
                            "confidence": 0.9,
                        })

                assay_idx = 0
                for target_entry in self._extract_target_entries(row.get("Target_Organism", ""), row.get("Activity", ""), max_activity_assays_per_record):
                    target_id = self.add_target_organism(target_entry["target"], evidence=evidence, properties={
                        "raw_target": target_entry["raw"],
                    })
                    if target_id:
                        self.add_relation(pid, "active_against", target_id, {
                            "source": "DRAMP",
                            "database_record": dramp_id,
                            "target_level": "organism",
                            "confidence": 0.85 if target_entry.get("measurement") else 0.75,
                        })
                    for target_class in target_entry.get("target_classes") or []:
                        target_class_id = self.add_target_class(target_class, evidence=evidence)
                        if target_class_id:
                            if target_id:
                                self.add_relation(target_id, "associated_with", target_class_id, {"source": "DRAMP"})
                            self.add_relation(pid, "active_against", target_class_id, {
                                "source": "DRAMP",
                                "database_record": dramp_id,
                                "target_level": "class",
                                "confidence": 0.8,
                            })
                    if target_entry.get("measurement"):
                        assay_id = self.add_assay("activity", dramp_id, assay_idx, properties={
                            "activity": self._clean_text(row.get("Activity")),
                            "target": target_entry["target"],
                            "measurement": target_entry["measurement"],
                            "raw": target_entry["raw"],
                        }, evidence=evidence)
                        stats["activity_assays"] += 1
                        self.add_relation(pid, "has_activity_assay", assay_id, {"source": "DRAMP", "database_record": dramp_id})
                        if target_id:
                            self.add_relation(assay_id, "active_against", target_id, {"source": "DRAMP"})
                        assay_idx += 1

                tox_count = self._enrich_toxicity_from_text(
                    pid, dramp_id, "hemolytic", row.get("Hemolytic_activity", ""), evidence, max_toxicity_assays_per_record
                )
                tox_count += self._enrich_toxicity_from_text(
                    pid, dramp_id, "cytotoxicity", row.get("Cytotoxicity", ""), evidence, max_toxicity_assays_per_record
                )
                stats["toxicity_assays"] += tox_count

                for feature_label, feature_props in self._structure_features_from_dramp(row):
                    feature_id = self.add_structure_feature(feature_label, properties=feature_props, evidence=evidence)
                    if feature_id:
                        stats["structure_features"] += 1
                        self.add_relation(pid, "has_structure", feature_id, {
                            "source": "DRAMP",
                            "database_record": dramp_id,
                        })

        self.enrichment_summary.update(stats)
        return stats

    def _enrich_toxicity_from_text(self, peptide_id: str, record_id: str, assay_kind: str,
                                   text: str, evidence: List[dict], max_assays: int) -> int:
        raw = self._clean_text(text)
        lower = raw.lower()
        if self._is_missing_text(raw) or "no information" in lower or "no cytotoxicity information" in lower or "no hemolysis information" in lower:
            return 0

        snippets = [s.strip() for s in re.split(r";|\.\s+|##", raw) if s.strip()]
        if not snippets:
            snippets = [raw]
        count = 0
        for idx, snippet in enumerate(snippets[:max_assays]):
            measurement = self._extract_measurement(snippet)
            observation = None
            if not measurement and re.search(r"\b(no|non)\s+[- ]?(hemolytic|cytotoxic|toxicity)", snippet, re.IGNORECASE):
                observation = "no_detected_toxicity"
            if not measurement and not observation:
                if idx > 0:
                    continue
                observation = "toxicity_reported_text"
            assay_id = self.add_assay("toxicity", f"{record_id}:{assay_kind}", idx, properties={
                "toxicity_kind": assay_kind,
                "measurement": measurement,
                "observation": observation,
                "raw": snippet,
            }, evidence=evidence)
            self.add_relation(peptide_id, "has_toxicity_assay", assay_id, {
                "source": "DRAMP",
                "database_record": record_id,
                "toxicity_kind": assay_kind,
            })
            count += 1
        return count

    def _structure_features_from_dramp(self, row: dict) -> List[Tuple[str, dict]]:
        features = []
        for key, label in [
            ("Structure", "secondary_structure"),
            ("Linear/Cyclic/Branched", "topology"),
            ("N-terminal_Modification", "n_terminal_modification"),
            ("C-terminal_Modification", "c_terminal_modification"),
            ("Other_Modifications", "other_modification"),
            ("Stereochemistry", "stereochemistry"),
            ("PDB_ID", "pdb_structure"),
        ]:
            value = self._clean_text(row.get(key))
            if value and not self._is_missing_text(value):
                features.append((f"{label}:{value[:120]}", {
                    "feature_type": label,
                    "value": value,
                    "database": "DRAMP",
                    "database_id": row.get("DRAMP_ID"),
                }))
        return features

    def enrich_from_dbaasp_index(self, data_dir: str) -> Dict[str, int]:
        """Enrich peptide KG from downloaded DBAASP /peptides index JSON pages."""
        stats = {
            "dbaasp_items_seen": 0,
            "dbaasp_sequences_matched": 0,
            "database_records": 0,
            "structure_features": 0,
            "modifications": 0,
        }
        if not data_dir or not os.path.isdir(data_dir):
            return stats
        for file_name in sorted(os.listdir(data_dir)):
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(data_dir, file_name)
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("data", []):
                stats["dbaasp_items_seen"] += 1
                for seq, record in self._dbaasp_sequences(item):
                    pid = f"peptide:{seq}"
                    if pid not in self.entities:
                        continue
                    stats["dbaasp_sequences_matched"] += 1
                    dbaasp_id = str(item.get("dbaaspId") or record.get("dbaaspId") or item.get("id") or record.get("id"))
                    evidence = [Evidence(
                        source="DBAASP",
                        reference_id=dbaasp_id,
                        url="https://dbaasp.org/peptides",
                        confidence=0.9,
                    ).to_dict()]
                    db_record_id = self.add_database_record(
                        dbaasp_id,
                        database="DBAASP",
                        properties={
                            "sequence": seq,
                            "name": self._clean_text(record.get("name") or item.get("name")),
                            "synthesis_type": record.get("synthesisType") or item.get("synthesisType"),
                            "complexity": record.get("complexity") or item.get("complexity"),
                            "pdb": record.get("pdb") or item.get("pdb"),
                            "pubchem_cid": record.get("pubchemCid") or item.get("pubchemCid"),
                            "n_terminus": record.get("nTerminus"),
                            "c_terminus": record.get("cTerminus"),
                        },
                        evidence=evidence,
                    )
                    stats["database_records"] += 1
                    self.add_relation(pid, "mapped_to", db_record_id, {"source": "DBAASP", "confidence": 0.9})
                    for term_key in ["nTerminus", "cTerminus"]:
                        term_value = record.get(term_key) or item.get(term_key)
                        if term_value and not self._is_missing_text(term_value):
                            mod_id = self._safe_id("modification", f"{term_key}:{term_value}")
                            self.add_entity(mod_id, "modification", {
                                "position": "N-terminal" if term_key == "nTerminus" else "C-terminal",
                                "value": term_value,
                            }, name=f"{term_key}:{term_value}", evidence=evidence)
                            stats["modifications"] += 1
                            self.add_relation(pid, "has_modification", mod_id, {"source": "DBAASP"})
                    pdb_value = record.get("pdb") or item.get("pdb")
                    if pdb_value:
                        for pdb_part in str(pdb_value).split("$$"):
                            pdb_id = pdb_part.split("$")[0].strip()
                            if pdb_id:
                                feature_id = self.add_structure_feature(f"pdb_structure:{pdb_id}", {
                                    "feature_type": "pdb_structure",
                                    "pdb_id": pdb_id,
                                    "raw": pdb_part,
                                    "database": "DBAASP",
                                }, evidence=evidence)
                                if feature_id:
                                    stats["structure_features"] += 1
                                    self.add_relation(pid, "has_structure", feature_id, {"source": "DBAASP"})
        self.enrichment_summary.update(stats)
        return stats

    def _dbaasp_sequences(self, item: dict) -> List[Tuple[str, dict]]:
        pairs = []
        seq = self._normalize_sequence(str(item.get("sequence") or ""))
        if self._is_standard_peptide(seq):
            pairs.append((seq, item))
        for monomer in item.get("monomers") or []:
            monomer_seq = self._normalize_sequence(str(monomer.get("sequence") or ""))
            if self._is_standard_peptide(monomer_seq):
                pairs.append((monomer_seq, monomer))
        return pairs

    def query_entity(self, entity_id: str) -> Optional[dict]:
        return self.entities.get(entity_id)

    def query_relations(self, head: Optional[str] = None, relation_type: Optional[str] = None, tail: Optional[str] = None) -> List[Tuple[str, str, str, dict]]:
        results = []
        for h, r, t, p in self.relations:
            if head and h != head:
                continue
            if relation_type and r != relation_type:
                continue
            if tail and t != tail:
                continue
            results.append((h, r, t, p))
        return results

    def get_neighbors(self, entity_id: str, max_depth: int = 1) -> Dict[str, List[dict]]:
        neighbors = {"outgoing": [], "incoming": []}
        for h, r, t, p in self.relations:
            if h == entity_id:
                neighbors["outgoing"].append({"target": t, "relation": r, "properties": p})
            if t == entity_id:
                neighbors["incoming"].append({"source": h, "relation": r, "properties": p})
        return neighbors

    def build_default_amp_knowledge_graph(self):
        amp_properties = {
            "length_range": {"type": "property", "properties": {"range": "12-20", "description": "Typical AMP length range"}},
            "positive_charge": {"type": "property", "properties": {"value": "high", "description": "High net positive charge (+2 to +9)"}},
            "amphipathic": {"type": "property", "properties": {"value": "yes", "description": "Amphipathic structure with hydrophobic and hydrophilic faces"}},
            "hydrophobicity": {"type": "property", "properties": {"value": "moderate", "description": "Moderate hydrophobicity (40-60%)"}},
            "low_toxicity": {"type": "property", "properties": {"value": "low", "description": "Low hemolytic activity and cytotoxicity"}},
        }
        for eid, props in amp_properties.items():
            self.add_entity(eid, "property", props["properties"])

        go_terms = {
            "GO_membrane_disruption": {"type": "go_term", "properties": {"id": "GO:0019835", "description": "Cytolysis, disruption of cell membrane"}},
            "GO_antimicrobial": {"type": "go_term", "properties": {"id": "GO:0019730", "description": "Antimicrobial humoral response"}},
            "GO_immune_modulation": {"type": "go_term", "properties": {"id": "GO:0050776", "description": "Regulation of immune response"}},
        }
        for eid, props in go_terms.items():
            self.add_entity(eid, "go_term", props["properties"])

        pathogens = {
            "gram_negative": {"type": "pathogen", "properties": {"class": "Gram-negative", "examples": "E. coli, P. aeruginosa", "membrane": "outer membrane + inner membrane"}},
            "gram_positive": {"type": "pathogen", "properties": {"class": "Gram-positive", "examples": "S. aureus, B. subtilis", "membrane": "single membrane + thick peptidoglycan"}},
        }
        for eid, props in pathogens.items():
            self.add_entity(eid, "pathogen", props["properties"])

        motifs = {
            "motif_KR": {"type": "motif", "properties": {"pattern": "K, R rich", "description": "Lysine and Arginine rich regions for membrane interaction"}},
            "motif_L": {"type": "motif", "properties": {"pattern": "L rich", "description": "Leucine rich for hydrophobic core formation"}},
            "motif_amphipathic_helix": {"type": "motif", "properties": {"pattern": "helix-forming", "description": "Amphipathic alpha-helix forming motif"}},
        }
        for eid, props in motifs.items():
            self.add_entity(eid, "motif", props["properties"])

        self.add_relation("motif_KR", "associated_with", "positive_charge")
        self.add_relation("motif_amphipathic_helix", "associated_with", "amphipathic")
        self.add_relation("GO_membrane_disruption", "related_to", "gram_negative")
        self.add_relation("GO_membrane_disruption", "related_to", "gram_positive")
        self.add_relation("GO_antimicrobial", "related_to", "gram_negative")
        self.add_relation("low_toxicity", "constrains", "hydrophobicity")
        self.add_relation("positive_charge", "promotes", "GO_membrane_disruption")
        self.add_relation("amphipathic", "promotes", "GO_membrane_disruption")
        self.add_relation("length_range", "constrains", "motif_amphipathic_helix")

        low_tox_rule = self.add_design_rule(
            "broad_spectrum_low_toxicity",
            "Broad-spectrum low-toxicity AMPs are usually short, cationic, moderately hydrophobic, and amphipathic.",
            {
                "target": "broad_spectrum",
                "recommended_length": [12, 30],
                "recommended_charge": [2, 9],
                "recommended_hydrophobic_ratio": [0.35, 0.60],
                "avoid": ["hydrophobic_ratio > 0.65", "long W/F stretches"],
                "motifs": ["K/R-rich", "amphipathic helix"],
            },
        )
        self.add_relation(low_tox_rule, "constrains", "length_range")
        self.add_relation(low_tox_rule, "suggests", "positive_charge")
        self.add_relation(low_tox_rule, "suggests", "amphipathic")
        self.add_relation(low_tox_rule, "warns_against", "hydrophobicity", {
            "warning": "Very high hydrophobicity may increase hemolysis/cytotoxicity",
        })

    def save(self, path: str = None):
        save_path = path or os.path.join(self.data_dir, "knowledge_graph.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "entities": self.entities,
            "relations": self.relations,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return save_path

    def load(self, path: str = None):
        load_path = path or os.path.join(self.data_dir, "knowledge_graph.json")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Knowledge graph file not found: {load_path}")
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = data["entities"]
        self.relations = [tuple(r) for r in data["relations"]]
        self._relation_keys = {
            (h, r, t, json.dumps(p or {}, ensure_ascii=False, sort_keys=True, default=str))
            for h, r, t, p in self.relations
        }

    def save_jsonl(self, data_dir: Optional[str] = None) -> Dict[str, str]:
        out_dir = data_dir or self.data_dir
        os.makedirs(out_dir, exist_ok=True)
        entity_path = os.path.join(out_dir, "entities.jsonl")
        relation_path = os.path.join(out_dir, "relations.jsonl")
        with open(entity_path, "w", encoding="utf-8") as f:
            for entity in self.entities.values():
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")
        with open(relation_path, "w", encoding="utf-8") as f:
            for head, relation, tail, properties in self.relations:
                f.write(json.dumps({
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "properties": properties,
                }, ensure_ascii=False) + "\n")
        return {"entities": entity_path, "relations": relation_path}

    def load_jsonl(self, data_dir: Optional[str] = None):
        in_dir = data_dir or self.data_dir
        entity_path = os.path.join(in_dir, "entities.jsonl")
        relation_path = os.path.join(in_dir, "relations.jsonl")
        self.entities = {}
        self.relations = []
        with open(entity_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entity = json.loads(line)
                self.entities[entity["id"]] = entity
        with open(relation_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rel = json.loads(line)
                self.relations.append((rel["head"], rel["relation"], rel["tail"], rel.get("properties", {})))
        self._relation_keys = {
            (h, r, t, json.dumps(p or {}, ensure_ascii=False, sort_keys=True, default=str))
            for h, r, t, p in self.relations
        }

    def summary(self) -> dict:
        entity_types = {}
        for e in self.entities.values():
            t = e["type"]
            entity_types[t] = entity_types.get(t, 0) + 1
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
        }

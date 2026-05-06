import json
import math
import os
from typing import Dict, List, Optional

from krpg.knowledge.knowledge_graph import KnowledgeGraph


class RAGRetriever:
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base")
        if knowledge_graph is not None:
            self.kg = knowledge_graph
        else:
            self.kg = KnowledgeGraph(data_dir=self.data_dir)
            entity_path = os.path.join(self.data_dir, "entities.jsonl")
            relation_path = os.path.join(self.data_dir, "relations.jsonl")
            if os.path.exists(entity_path) and os.path.exists(relation_path):
                self.kg.load_jsonl(self.data_dir)
        self._knowledge_templates: Dict[str, str] = {}
        self._knowledge_entries: List[Dict] = []

    def load_default_templates(self):
        self._knowledge_templates = {
            "broad_spectrum_low_toxicity": (
                "Design an antimicrobial peptide with broad-spectrum activity, "
                "low toxicity to mammalian cells, moderate length (12-20 aa), "
                "high positive charge, and amphipathic structure."
            ),
            "gram_negative_targeted": (
                "Design an antimicrobial peptide targeting Gram-negative bacteria. "
                "Focus on membrane disruption mechanism, high positive charge, "
                "and amphipathic alpha-helix formation."
            ),
            "gram_positive_targeted": (
                "Design an antimicrobial peptide targeting Gram-positive bacteria. "
                "Consider thick peptidoglycan layer penetration and membrane disruption."
            ),
            "low_toxicity_high_stability": (
                "Design an antimicrobial peptide with very low hemolytic toxicity "
                "and high proteolytic stability. Moderate antimicrobial activity is acceptable."
            ),
        }

    def _build_knowledge_entries(self):
        """Build searchable knowledge entries from KG entities and relations."""
        expected_entries = len(self.kg.entities) + len(self.kg.relations)
        if self._knowledge_entries and len(self._knowledge_entries) == expected_entries:
            return
        self._knowledge_entries = []
        for eid, entity in self.kg.entities.items():
            props = entity.get("properties", {})
            name = entity.get("name", eid)
            desc = props.get("description", "") or props.get("value", "") or str(props)
            self._knowledge_entries.append({
                "id": eid,
                "type": entity["type"],
                "content": f"{name}: {desc}",
                "keywords": self._tokenize(f"{eid} {name} {desc}"),
                "entity": entity,
            })
        for h, r, t, props in self.kg.relations:
            content = f"{h} {r} {t}"
            self._knowledge_entries.append({
                "id": f"{h}->{t}",
                "type": "relation",
                "content": content,
                "keywords": self._tokenize(content),
                "relation": {"head": h, "relation": r, "tail": t, "properties": props},
            })

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization for similarity computation."""
        return text.lower().replace("-", " ").replace("_", " ").split()

    @staticmethod
    def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        set_a = set(tokens_a)
        set_b = set(tokens_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def retrieve_by_target(self, target: str, constraints: Dict[str, str] = None) -> Dict[str, List[str]]:
        results = {
            "properties": [],
            "go_terms": [],
            "pathogens": [],
            "motifs": [],
            "relations": [],
        }

        query_tokens = self._tokenize(target)

        self._build_knowledge_entries()
        scored_entries = []
        for entry in self._knowledge_entries:
            sim = self._jaccard_similarity(query_tokens, entry["keywords"])
            if sim > 0:
                scored_entries.append((sim, entry))
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        for sim, entry in scored_entries[:10]:
            etype = entry["type"]
            category_map = {
                "property": "properties",
                "go_term": "go_terms",
                "pathogen": "pathogens",
                "motif": "motifs",
                "relation": "relations",
            }
            category = category_map.get(etype, "properties")
            results[category].append(f"[KG] {entry['content']} (sim={sim:.2f})")

        if "broad" in target.lower() or "广谱" in target:
            results["go_terms"].append("GO:0019835 - Cytolysis, membrane disruption")
            results["pathogens"].append("Gram-negative bacteria (E. coli, P. aeruginosa)")
            results["pathogens"].append("Gram-positive bacteria (S. aureus, B. subtilis)")

        if "gram-negative" in target.lower() or "革兰氏阴性" in target:
            results["pathogens"].append("Gram-negative: outer membrane target, LPS interaction")
            results["properties"].append("High positive charge (+4 to +8) for LPS binding")

        if "gram-positive" in target.lower() or "革兰氏阳性" in target:
            results["pathogens"].append("Gram-positive: peptidoglycan target, membrane disruption")
            results["properties"].append("Moderate positive charge (+2 to +5)")

        if "low toxicity" in target.lower() or "低毒" in target:
            results["properties"].append("Low hemolytic activity: avoid high hydrophobicity (>60%)")
            results["properties"].append("Moderate hydrophobicity (40-50%) for selectivity")
            results["motifs"].append("Balance charged and hydrophobic residues")

        if "amphipathic" in target.lower() or "两亲" in target:
            results["properties"].append("Amphipathic structure: polar and non-polar faces")
            results["motifs"].append("Alpha-helix forming: (K/R)-X-X-(L/I/V)-X pattern")

        if "positive charge" in target.lower() or "正电荷" in target:
            results["properties"].append("Net positive charge: +2 to +9 typical for AMPs")
            results["motifs"].append("K and R rich regions for membrane interaction")

        if constraints:
            if "length" in constraints:
                results["properties"].append(f"Length constraint: {constraints['length']} amino acids")
            if "charge" in constraints:
                results["properties"].append(f"Charge constraint: net charge {constraints['charge']}")
            if "toxicity" in constraints:
                results["properties"].append(f"Toxicity constraint: {constraints['toxicity']}")

        kg_results = self.kg.get_neighbors("positive_charge")
        if kg_results["outgoing"]:
            results["relations"].append("Positive charge promotes membrane disruption (GO:0019835)")

        return results

    @staticmethod
    def _append_unique(items: List, value):
        if value not in items:
            items.append(value)

    @staticmethod
    def _parse_range_from_text(text: str, default=None):
        import re
        nums = [float(n) for n in re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")]
        if len(nums) >= 2:
            return [nums[0], nums[1]]
        if len(nums) == 1:
            return [nums[0], nums[0]]
        return default

    def _collect_peptide_stats(self) -> Dict:
        peptides = [e for e in self.kg.entities.values() if e.get("type") == "peptide"]
        if not peptides:
            return {}

        def prop_values(name: str):
            values = []
            for peptide in peptides:
                value = peptide.get("properties", {}).get(name)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            return values

        def quantile(values: List[float], q: float):
            if not values:
                return None
            values = sorted(values)
            idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
            return values[idx]

        stats = {"n_peptides": len(peptides)}
        for key in ["length", "net_charge", "hydrophobic_ratio", "aromatic_ratio", "isoelectric_point"]:
            values = prop_values(key)
            if values:
                stats[key] = {
                    "median": round(quantile(values, 0.5), 3),
                    "q25": round(quantile(values, 0.25), 3),
                    "q75": round(quantile(values, 0.75), 3),
                }
        stats["examples"] = [
            {
                "sequence": p.get("properties", {}).get("sequence", p.get("name")),
                "length": p.get("properties", {}).get("length"),
                "net_charge": p.get("properties", {}).get("net_charge"),
                "hydrophobic_ratio": p.get("properties", {}).get("hydrophobic_ratio"),
            }
            for p in peptides[:5]
        ]
        return stats

    def retrieve_by_design_spec(self, design_spec: Dict, top_k: int = 8) -> Dict[str, List]:
        """Return structured, prompt-ready evidence for a design specification.

        This is the report-aligned RAG interface: it keeps evidence categories
        separate so PromptBuilder, PromptEncoder, and output provenance can use
        the same retrieval result without parsing free text.
        """
        target = design_spec.get("target", "")
        activity = design_spec.get("activity", "")
        constraint = design_spec.get("constraint", "")
        preference = design_spec.get("preference", "")
        query = ", ".join(part for part in [target, activity, constraint, preference] if part)

        legacy = self.retrieve_by_target(query, design_spec.get("constraints"))
        context = {
            "target_context": [],
            "activity_evidence": [],
            "property_constraints": [],
            "motif_hints": [],
            "toxicity_warnings": [],
            "mechanism_hints": [],
            "design_rules": [],
            "citations": [],
        }

        for item in legacy.get("pathogens", []):
            self._append_unique(context["target_context"], {"text": item, "source": "kg_or_rule"})
        for item in legacy.get("go_terms", []):
            self._append_unique(context["mechanism_hints"], {"text": item, "source": "kg_or_rule"})
        for item in legacy.get("properties", []):
            self._append_unique(context["property_constraints"], {"text": item, "source": "kg_or_rule"})
        for item in legacy.get("motifs", []):
            self._append_unique(context["motif_hints"], {"text": item, "source": "kg_or_rule"})
        for item in legacy.get("relations", []):
            self._append_unique(context["mechanism_hints"], {"text": item, "source": "kg_relation"})

        peptide_stats = self._collect_peptide_stats()
        if peptide_stats:
            context["activity_evidence"].append({
                "text": f"Loaded {peptide_stats['n_peptides']} peptide records for AMP distribution statistics.",
                "source": "peptide_kg",
                "stats": peptide_stats,
            })
            for key, label in [
                ("length", "Length"),
                ("net_charge", "Net charge"),
                ("hydrophobic_ratio", "Hydrophobic ratio"),
            ]:
                if key in peptide_stats:
                    s = peptide_stats[key]
                    context["property_constraints"].append({
                        "text": f"{label} empirical middle range: {s['q25']} to {s['q75']} (median {s['median']}).",
                        "source": "peptide_kg_statistics",
                        "feature": key,
                        "range": [s["q25"], s["q75"]],
                    })
            for example in peptide_stats.get("examples", []):
                context["activity_evidence"].append({
                    "text": f"AMP example {example['sequence']} length={example.get('length')} charge={example.get('net_charge')}.",
                    "source": "peptide_kg_example",
                    "sequence": example.get("sequence"),
                })

        self._build_knowledge_entries()
        scored_entries = []
        query_tokens = self._tokenize(query)
        for entry in self._knowledge_entries:
            sim = self._jaccard_similarity(query_tokens, entry["keywords"])
            if sim > 0:
                scored_entries.append((sim, entry))
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        for sim, entry in scored_entries[:top_k]:
            if entry["type"] == "design_rule":
                entity = entry.get("entity", {})
                props = entity.get("properties", {})
                context["design_rules"].append({
                    "id": entity.get("id", entry["id"]),
                    "text": props.get("description", entry["content"]),
                    "properties": props,
                    "relevance": round(sim, 4),
                    "source": "knowledge_graph",
                })
                for motif in props.get("motifs", []):
                    self._append_unique(context["motif_hints"], {"text": motif, "source": "design_rule"})
                for warning in props.get("avoid", []):
                    self._append_unique(context["toxicity_warnings"], {"text": warning, "source": "design_rule"})
            elif entry["type"] in {"go_term", "mechanism"}:
                self._append_unique(context["mechanism_hints"], {
                    "text": entry["content"],
                    "source": entry["type"],
                    "relevance": round(sim, 4),
                })

        length_range = design_spec.get("length_range") or self._parse_range_from_text(preference)
        if length_range:
            context["property_constraints"].append({
                "text": f"User/requested length range: {length_range[0]}-{length_range[1]} aa.",
                "source": "design_spec",
                "feature": "length",
                "range": length_range,
            })
        charge_range = design_spec.get("charge_range") or self._parse_range_from_text(
            design_spec.get("charge", "") or preference,
            default=None,
        )
        if charge_range and ("charge" in preference.lower() or "电荷" in preference):
            context["property_constraints"].append({
                "text": f"Requested net charge range: {charge_range[0]} to {charge_range[1]}.",
                "source": "design_spec",
                "feature": "net_charge",
                "range": charge_range,
            })

        if "low toxicity" in constraint.lower() or "低毒" in constraint:
            context["toxicity_warnings"].append({
                "text": "Keep hydrophobicity moderate and avoid long W/F-rich stretches to reduce hemolysis risk.",
                "source": "design_spec",
            })

        return context

    def format_knowledge_block(self, retrieved: Dict[str, List[str]]) -> str:
        lines = ["Retrieved Knowledge:"]
        for category, items in retrieved.items():
            if items:
                lines.append(f"\n- {category.replace('_', ' ').title()}:")
                for item in items:
                    lines.append(f"  * {item}")
        return "\n".join(lines)

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[dict]:
        query_lower = query.lower()
        query_tokens = self._tokenize(query)

        results = []

        self._build_knowledge_entries()
        for entry in self._knowledge_entries:
            sim = self._jaccard_similarity(query_tokens, entry["keywords"])
            if sim > 0:
                results.append({
                    "source": entry["type"],
                    "content": entry["content"],
                    "relevance": round(sim, 4),
                })

        kg_summary = self.kg.summary()
        results.append({
            "source": "knowledge_graph",
            "content": f"Knowledge graph contains {kg_summary['total_entities']} entities and {kg_summary['total_relations']} relations",
            "relevance": 0.9 if any(kw in query_lower for kw in ["amp", "peptide", "antimicrobial", "抗菌"]) else 0.5,
        })

        if "toxicity" in query_lower or "toxic" in query_lower or "毒" in query_lower:
            results.append({
                "source": "property",
                "content": "Low toxicity AMPs typically have moderate hydrophobicity (40-50%), balanced charge distribution, and avoid highly hydrophobic stretches.",
                "relevance": 0.95,
            })

        if "gram" in query_lower or "negative" in query_lower or "阴性" in query_lower:
            results.append({
                "source": "pathogen",
                "content": "Gram-negative bacteria have an outer membrane rich in LPS. AMPs targeting Gram-negatives often have high positive charge for LPS binding.",
                "relevance": 0.9,
            })

        if "stability" in query_lower or "稳定" in query_lower:
            results.append({
                "source": "property",
                "content": "Peptide stability can be enhanced by: (1) including D-amino acids, (2) cyclization, (3) reducing protease-susceptible motifs (e.g., RR, KK), (4) increasing hydrophobic content.",
                "relevance": 0.85,
            })

        if "helix" in query_lower or "螺旋" in query_lower or "amphipathic" in query_lower or "两亲" in query_lower:
            results.append({
                "source": "structure",
                "content": "Amphipathic alpha-helices are a common AMP structure. Key pattern: polar (K/R) and non-polar (L/I/V/F) residues on opposite faces of the helix.",
                "relevance": 0.9,
            })

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

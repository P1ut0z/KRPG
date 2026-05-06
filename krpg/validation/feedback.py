import json
from typing import Dict, List, Optional


class FeedbackOptimizer:
    def __init__(self):
        self.history: List[Dict] = []

    def analyze_results(self, results: List[Dict]) -> Dict:
        if not results:
            return {"patterns": [], "suggestions": [], "avg_amp_score": 0.0,
                    "avg_toxicity_score": 0.0, "avg_stability_score": 0.0,
                    "n_high_scoring": 0, "n_low_toxic": 0}

        avg_amp = sum(r.get("amp_score", 0) for r in results) / len(results)
        avg_tox = sum(r.get("toxicity_score", 0) for r in results) / len(results)
        avg_stab = sum(r.get("stability_score", 0) for r in results) / len(results)

        high_scoring = [r for r in results if r.get("amp_score", 0) > 0.6]
        low_toxic = [r for r in results if r.get("toxicity_score", 0) < 0.4]

        patterns = self._extract_patterns(high_scoring)
        suggestions = []

        if avg_amp < 0.4:
            suggestions.append("Increase positive charge (more K/R residues)")
            suggestions.append("Increase hydrophobic residue ratio")
        if avg_tox > 0.5:
            suggestions.append("Reduce hydrophobicity to lower toxicity")
            suggestions.append("Reduce tryptophan and phenylalanine content")
        if avg_stab < 0.4:
            suggestions.append("Add cysteine residues for potential disulfide bonds")
            suggestions.append("Reduce glycine and proline content")
        if len(high_scoring) < len(results) * 0.3:
            suggestions.append("Loosen constraints to allow more diverse sequences")
        if len(low_toxic) < len(results) * 0.5:
            suggestions.append("Strengthen low-toxicity constraint in prompt")

        return {
            "patterns": patterns,
            "suggestions": suggestions,
            "avg_amp_score": round(avg_amp, 4),
            "avg_toxicity_score": round(avg_tox, 4),
            "avg_stability_score": round(avg_stab, 4),
            "n_high_scoring": len(high_scoring),
            "n_low_toxic": len(low_toxic),
        }

    def _extract_patterns(self, sequences: List[Dict]) -> List[str]:
        if not sequences:
            return []
        patterns = []
        all_seqs = [s.get("sequence", "").upper() for s in sequences if s.get("sequence")]

        if not all_seqs:
            return []

        n_term_chars = [s[0] for s in all_seqs if s]
        if n_term_chars:
            common_n = max(set(n_term_chars), key=n_term_chars.count)
            if n_term_chars.count(common_n) > len(n_term_chars) * 0.3:
                patterns.append(f"N-terminal {common_n} appears frequently")

        all_charged = sum(s.count("K") + s.count("R") for s in all_seqs)
        all_len = sum(len(s) for s in all_seqs)
        if all_len > 0 and all_charged / all_len > 0.2:
            patterns.append("High K/R content correlates with high AMP score")

        all_hydro = sum(s.count("L") + s.count("I") + s.count("V") + s.count("F") + s.count("W") for s in all_seqs)
        if all_len > 0 and all_hydro / all_len > 0.3:
            patterns.append("Moderate hydrophobic content correlates with high AMP score")

        return patterns

    def generate_feedback(self, results: List[Dict], round_num: int = 1) -> Dict:
        analysis = self.analyze_results(results)

        feedback = {
            "round": round_num,
            "analysis": analysis,
            "updated_constraints": {},
        }

        if analysis["avg_amp_score"] < 0.4:
            feedback["updated_constraints"]["charge"] = "increase positive charge (target: +4 to +8)"
            feedback["updated_constraints"]["hydrophobicity"] = "increase to 40-55%"
        if analysis["avg_toxicity_score"] > 0.5:
            feedback["updated_constraints"]["toxicity"] = "reduce hydrophobicity, avoid W/WW motifs"
        if analysis["avg_stability_score"] < 0.4:
            feedback["updated_constraints"]["stability"] = "add C residues, reduce G/P content"

        self.history.append(feedback)
        return feedback

    def apply_feedback_to_spec(self, original_spec: Dict, feedback: Dict) -> Dict:
        """Apply feedback to produce an updated design specification for the next round.

        This closes the loop: validation results -> constraint updates -> new prompt.
        """
        updated_spec = dict(original_spec)
        constraints = feedback.get("updated_constraints", {})
        analysis = feedback.get("analysis", {})

        preference_parts = []
        existing_pref = updated_spec.get("preference", "")
        if existing_pref:
            preference_parts = [p.strip() for p in existing_pref.split(",")]

        if "charge" in constraints:
            charge_val = "Positive Charge +4 to +8"
            replaced = False
            for i, p in enumerate(preference_parts):
                if "charge" in p.lower() or "电荷" in p:
                    preference_parts[i] = charge_val
                    replaced = True
                    break
            if not replaced:
                preference_parts.append(charge_val)

        if "hydrophobicity" in constraints:
            hydro_val = "Moderate Hydrophobicity 40-55%"
            preference_parts.append(hydro_val)

        if "toxicity" in constraints:
            constraint_text = updated_spec.get("constraint", "")
            if "low toxicity" not in constraint_text.lower():
                updated_spec["constraint"] = f"{constraint_text}, Very Low Toxicity".strip(", ")

        if "stability" in constraints:
            preference_parts.append("Include Cysteine for Stability")

        updated_spec["preference"] = ", ".join(preference_parts)

        if analysis.get("patterns"):
            updated_spec["feedback_patterns"] = analysis["patterns"]

        return updated_spec

    def get_optimized_prompt_modifications(self, feedback: Dict) -> str:
        mods = []
        constraints = feedback.get("updated_constraints", {})
        for key, value in constraints.items():
            mods.append(f"- {key}: {value}")
        suggestions = feedback.get("analysis", {}).get("suggestions", [])
        for s in suggestions:
            mods.append(f"- {s}")
        return "\n".join(mods) if mods else "No modifications needed."

    def save_history(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def load_history(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.history = json.load(f)

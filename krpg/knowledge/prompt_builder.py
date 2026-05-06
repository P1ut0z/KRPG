from typing import Dict, List, Optional


class PromptBuilder:
    def __init__(self):
        self.template = (
            "Target: {target}\n"
            "Activity Requirement: {activity}\n"
            "Hard Constraints: {constraint}\n"
            "Soft Preferences: {preference}\n"
            "{knowledge_block}\n"
            "Generation Instruction: Generate antimicrobial peptide candidate sequences that satisfy the constraints above."
        )

    @staticmethod
    def _format_item(item) -> str:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("sequence") or str(item)
            source = item.get("source")
            relevance = item.get("relevance")
            suffix = []
            if source:
                suffix.append(f"source={source}")
            if relevance is not None:
                suffix.append(f"relevance={relevance}")
            return f"{text} ({', '.join(suffix)})" if suffix else text
        return str(item)

    def _format_knowledge_block(self, retrieved_knowledge: Dict[str, List]) -> str:
        if not retrieved_knowledge:
            return ""

        section_names = {
            "target_context": "Target Context",
            "activity_evidence": "Activity Evidence",
            "property_constraints": "Property Constraints",
            "motif_hints": "Motif Hints",
            "toxicity_warnings": "Avoid / Toxicity Warnings",
            "mechanism_hints": "Mechanism Hints",
            "design_rules": "Design Rules",
            "citations": "Citations",
            "properties": "Property Constraints",
            "go_terms": "Mechanism Hints",
            "pathogens": "Target Context",
            "motifs": "Motif Hints",
            "relations": "KG Relations",
        }

        lines = ["\nRetrieved Knowledge:"]
        for category, items in retrieved_knowledge.items():
            if not items:
                continue
            title = section_names.get(category, category.replace("_", " ").title())
            lines.append(f"\n- {title}:")
            for item in items:
                lines.append(f"  * {self._format_item(item)}")
        return "\n".join(lines)

    def build_prompt(self, target: str, activity: str = "", constraint: str = "",
                     preference: str = "", retrieved_knowledge: Dict[str, List[str]] = None) -> str:
        knowledge_block = self._format_knowledge_block(retrieved_knowledge or {})

        return self.template.format(
            target=target,
            activity=activity,
            constraint=constraint,
            preference=preference,
            knowledge_block=knowledge_block,
        )

    def build_structured_prompt(self, design_spec: Dict[str, str]) -> str:
        target = design_spec.get("target", "Broad-Spectrum AMP")
        activity = design_spec.get("activity", "Gram-Negative")
        constraint = design_spec.get("constraint", "Low Toxicity")
        preference = design_spec.get("preference", "Length 12-20, Positive Charge, Amphipathic")

        retrieved = {}
        if "retrieved_context" in design_spec:
            retrieved = design_spec["retrieved_context"]
        elif "knowledge" in design_spec:
            retrieved = design_spec["knowledge"]

        return self.build_prompt(target, activity, constraint, preference, retrieved)

    def build_prompt_record(self, raw_user_prompt: str, design_spec: Dict,
                            retrieved_context: Dict) -> Dict:
        """Create a traceable prompt record for generation provenance."""
        spec = dict(design_spec)
        spec["retrieved_context"] = retrieved_context
        rendered_prompt = self.build_structured_prompt(spec)
        citations = retrieved_context.get("citations", []) if retrieved_context else []
        return {
            "raw_user_prompt": raw_user_prompt,
            "design_spec": design_spec,
            "retrieved_context": retrieved_context,
            "rendered_prompt": rendered_prompt,
            "citations": citations,
        }

    def build_multi_objective_prompt(self, objectives: List[Dict[str, str]]) -> str:
        parts = []
        for i, obj in enumerate(objectives, 1):
            prompt = self.build_structured_prompt(obj)
            parts.append(f"=== Objective {i} ===\n{prompt}")
        return "\n\n".join(parts)

    def build_feedback_prompt(self, original_prompt: str, feedback: Dict[str, List[str]],
                              high_scoring_patterns: List[str]) -> str:
        feedback_lines = ["\n[Feedback from Previous Round]"]
        for category, items in feedback.items():
            if items:
                feedback_lines.append(f"\n- {category}:")
                for item in items:
                    feedback_lines.append(f"  * {item}")

        if high_scoring_patterns:
            feedback_lines.append("\n- High-scoring patterns to emphasize:")
            for pattern in high_scoring_patterns:
                feedback_lines.append(f"  * {pattern}")

        feedback_block = "\n".join(feedback_lines)
        return original_prompt + feedback_block

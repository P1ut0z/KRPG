import re
from typing import Dict, List, Optional


class PromptParser:
    """Lightweight parser from user-facing prompt text to design_spec.

    This keeps the report's明文 Prompt interface usable without pretending the
    small sequence model directly understands long natural-language text.
    """

    def parse(self, text: str, defaults: Optional[Dict] = None) -> Dict:
        defaults = defaults or {}
        lower = text.lower()
        spec = {
            "target": defaults.get("target", "Broad-Spectrum AMP"),
            "activity": defaults.get("activity", ""),
            "constraint": defaults.get("constraint", ""),
            "preference": defaults.get("preference", ""),
            "raw_prompt": text,
        }

        if any(k in lower for k in ["gram-negative", "革兰氏阴性", "阴性菌"]):
            spec["activity"] = "Gram-Negative"
        elif any(k in lower for k in ["gram-positive", "革兰氏阳性", "阳性菌"]):
            spec["activity"] = "Gram-Positive"

        if any(k in lower for k in ["broad", "广谱"]):
            spec["target"] = "Broad-Spectrum AMP"
        if any(k in lower for k in ["low toxicity", "low-toxic", "低毒", "低毒性"]):
            spec["constraint"] = "Low Toxicity"
        if any(k in lower for k in ["stable", "stability", "稳定"]):
            spec["constraint"] = self._append_phrase(spec["constraint"], "High Stability")

        preference_parts: List[str] = []
        length_range = self._extract_length_range(text)
        if length_range:
            spec["length_range"] = length_range
            preference_parts.append(f"Length {int(length_range[0])}-{int(length_range[1])}")
        if any(k in lower for k in ["positive charge", "正电荷", "阳离子"]):
            preference_parts.append("Positive Charge")
            charge_range = self._extract_charge_range(text)
            if charge_range:
                spec["charge_range"] = charge_range
        if any(k in lower for k in ["amphipathic", "两亲"]):
            preference_parts.append("Amphipathic")
        if any(k in lower for k in ["hydrophobic", "疏水"]):
            preference_parts.append("Moderate Hydrophobicity")
        if preference_parts:
            spec["preference"] = ", ".join(preference_parts)

        novelty = self._extract_novelty_threshold(text)
        if novelty is not None:
            spec["novelty_threshold"] = novelty

        return spec

    @staticmethod
    def _append_phrase(existing: str, phrase: str) -> str:
        return phrase if not existing else f"{existing}, {phrase}"

    @staticmethod
    def _extract_length_range(text: str):
        patterns = [
            r"length\s*(\d+)\s*[-–]\s*(\d+)",
            r"长度\s*(\d+)\s*[-–到至]\s*(\d+)",
            r"(\d+)\s*[-–]\s*(\d+)\s*aa",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return [float(match.group(1)), float(match.group(2))]
        return None

    @staticmethod
    def _extract_charge_range(text: str):
        match = re.search(r"\+?\s*(\d+)\s*(?:to|[-–到至])\s*\+?\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        return None

    @staticmethod
    def _extract_novelty_threshold(text: str):
        match = re.search(r"(?:identity|similarity|相似性|新颖性)[^\d]*(\d+(?:\.\d+)?)\s*%?", text, flags=re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value / 100.0 if value > 1 else value

from typing import Dict, List


class ToxicityPredictor:
    def __init__(self):
        self.toxic_motifs = [
            "RR", "KK", "WW", "FF", "YY",
            "CCC", "PPP", "HHH",
        ]
        self.high_toxicity_aas = {"W": 2.0, "F": 1.5, "Y": 1.2, "M": 1.0, "I": 1.0, "L": 1.0}
        self.low_toxicity_aas = {"G": -0.5, "S": -0.3, "T": -0.3, "N": -0.2, "Q": -0.2}

    def predict(self, sequence: str) -> Dict:
        seq = sequence.upper()
        length = len(seq)
        if length == 0:
            return {"sequence": seq, "toxicity_score": 1.0, "is_toxic": True, "reason": "Empty sequence"}

        toxicity_score = 0.0
        reasons = []

        for motif in self.toxic_motifs:
            count = seq.count(motif)
            if count > 0:
                toxicity_score += count * 0.15
                reasons.append(f"Contains toxic motif '{motif}' ({count}x)")

        for aa, weight in self.high_toxicity_aas.items():
            count = seq.count(aa)
            toxicity_score += count * weight * 0.05

        for aa, weight in self.low_toxicity_aas.items():
            count = seq.count(aa)
            toxicity_score += count * weight * 0.03

        hydrophobic_ratio = sum(1 for aa in seq if aa in "AILMFWV") / length
        if hydrophobic_ratio > 0.6:
            toxicity_score += 0.3
            reasons.append(f"High hydrophobicity ({hydrophobic_ratio:.2f}) may increase toxicity")
        elif hydrophobic_ratio < 0.3:
            toxicity_score -= 0.1
            reasons.append(f"Low hydrophobicity ({hydrophobic_ratio:.2f}) suggests low toxicity")

        charge = sum(1 for aa in seq if aa in "KRH") - sum(1 for aa in seq if aa in "DE")
        if charge > 8:
            toxicity_score += 0.2
            reasons.append(f"Very high charge ({charge}) may increase toxicity")

        toxicity_score = max(0.0, min(1.0, toxicity_score))
        is_toxic = toxicity_score > 0.5

        return {
            "sequence": seq,
            "toxicity_score": round(toxicity_score, 4),
            "is_toxic": is_toxic,
            "details": {
                "hydrophobic_ratio": round(hydrophobic_ratio, 3),
                "net_charge": charge,
            },
            "reasons": reasons,
        }

    def batch_predict(self, sequences: List[str]) -> List[Dict]:
        return [self.predict(seq) for seq in sequences]

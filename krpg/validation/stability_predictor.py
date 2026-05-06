from typing import Dict, List


class StabilityPredictor:
    def __init__(self):
        self.destabilizing_motifs = [
            "DG", "DP", "GG", "PG", "PP",
            "NQ", "QT", "TS", "ST",
        ]
        self.protease_sites = [
            "RR", "RK", "KR", "KK",
            "FK", "YK", "WK",
        ]
        self.stabilizing_aas = {"C": 1.5, "W": 1.0, "Y": 0.8, "F": 0.8, "I": 0.6, "L": 0.6, "V": 0.5}
        self.destabilizing_aas = {"P": -1.0, "G": -0.5, "S": -0.3, "T": -0.3, "N": -0.3, "Q": -0.2}

    def predict(self, sequence: str) -> Dict:
        seq = sequence.upper()
        length = len(seq)
        if length == 0:
            return {"sequence": seq, "stability_score": 0.0, "is_stable": False, "reason": "Empty sequence"}

        stability_score = 0.5
        reasons = []

        for motif in self.destabilizing_motifs:
            count = seq.count(motif)
            if count > 0:
                stability_score -= count * 0.1
                reasons.append(f"Contains destabilizing motif '{motif}' ({count}x)")

        for motif in self.protease_sites:
            count = seq.count(motif)
            if count > 0:
                stability_score -= count * 0.08
                reasons.append(f"Contains protease site '{motif}' ({count}x)")

        for aa, weight in self.stabilizing_aas.items():
            count = seq.count(aa)
            stability_score += count * weight * 0.02

        for aa, weight in self.destabilizing_aas.items():
            count = seq.count(aa)
            stability_score += count * weight * 0.02

        glycine_count = seq.count("G")
        proline_count = seq.count("P")
        if glycine_count / length > 0.3:
            stability_score -= 0.15
            reasons.append(f"High glycine content ({glycine_count}/{length}) reduces stability")
        if proline_count / length > 0.2:
            stability_score -= 0.15
            reasons.append(f"High proline content ({proline_count}/{length}) may disrupt structure")

        cysteine_count = seq.count("C")
        if cysteine_count >= 2:
            stability_score += 0.2
            reasons.append(f"Potential disulfide bonds ({cysteine_count} Cys)")

        stability_score = max(0.0, min(1.0, stability_score))
        is_stable = stability_score >= 0.5

        return {
            "sequence": seq,
            "stability_score": round(stability_score, 4),
            "is_stable": is_stable,
            "details": {
                "length": length,
                "cysteine_count": cysteine_count,
                "glycine_ratio": round(glycine_count / length, 3),
                "proline_ratio": round(proline_count / length, 3),
            },
            "reasons": reasons,
        }

    def batch_predict(self, sequences: List[str]) -> List[Dict]:
        return [self.predict(seq) for seq in sequences]

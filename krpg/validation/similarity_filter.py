import math
from typing import Dict, List, Optional, Tuple


class SimilarityFilter:
    def __init__(self, known_amps: Optional[List[str]] = None):
        self.known_amps = known_amps or []

    def set_known_amps(self, sequences: List[str]):
        self.known_amps = sequences

    def sequence_identity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity with alignment-aware comparison.

        For equal-length sequences, uses direct positional comparison.
        For different lengths, uses sliding window best-match to find
        the highest local identity.
        """
        s1 = seq1.upper()
        s2 = seq2.upper()
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        if len1 == len2:
            matches = sum(1 for a, b in zip(s1, s2) if a == b)
            return matches / len1

        shorter, longer = (s1, s2) if len1 <= len2 else (s2, s1)
        short_len = len(shorter)
        long_len = len(longer)

        best_identity = 0.0
        for offset in range(long_len - short_len + 1):
            window = longer[offset:offset + short_len]
            matches = sum(1 for a, b in zip(shorter, window) if a == b)
            identity = matches / long_len
            best_identity = max(best_identity, identity)

        return best_identity

    def max_similarity_to_known(self, sequence: str) -> Tuple[float, Optional[str]]:
        if not self.known_amps:
            return 0.0, None
        max_sim = 0.0
        most_similar = None
        for known in self.known_amps:
            sim = self.sequence_identity(sequence, known)
            if sim > max_sim:
                max_sim = sim
                most_similar = known
        return max_sim, most_similar

    def filter_by_similarity(self, sequences: List[str], threshold: float = 0.8) -> List[Dict]:
        results = []
        for seq in sequences:
            max_sim, most_similar = self.max_similarity_to_known(seq)
            is_novel = max_sim < threshold
            results.append({
                "sequence": seq,
                "max_similarity": round(max_sim, 4),
                "most_similar_known": most_similar,
                "is_novel": is_novel,
            })
        return results

    def deduplicate(self, sequences: List[str], identity_threshold: float = 0.9) -> List[str]:
        unique = []
        for seq in sequences:
            is_dup = False
            for existing in unique:
                if self.sequence_identity(seq, existing) >= identity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(seq)
        return unique


class PhysicochemicalFilter:
    def __init__(self):
        self.molecular_weight_table = {
            "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
            "G": 75.07, "H": 155.16, "I": 131.18, "K": 146.19, "L": 131.18,
            "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
            "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
        }
        self.hydrophobicity_table = {
            "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
            "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
            "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
            "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
        }
        self.pka_table = {
            "N_term": 9.69, "C_term": 2.34,
            "K": 10.53, "R": 12.48, "H": 6.00,
            "D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07,
        }

    def compute_properties(self, sequence: str) -> Dict:
        seq = sequence.upper()
        length = len(seq)
        if length == 0:
            return {"length": 0, "molecular_weight": 0.0, "net_charge": 0,
                    "hydrophobicity": 0.0, "isoelectric_point": 0.0,
                    "positive_residues": 0, "negative_residues": 0,
                    "hydrophobic_ratio": 0.0, "aromatic_ratio": 0.0}

        mw = sum(self.molecular_weight_table.get(aa, 0) for aa in seq) - 18.015 * (length - 1)
        n_pos = sum(1 for aa in seq if aa in "KRH")
        n_neg = sum(1 for aa in seq if aa in "DE")
        net_charge = n_pos - n_neg
        avg_hydro = sum(self.hydrophobicity_table.get(aa, 0) for aa in seq) / length
        pI = self._compute_pI(seq)
        hydrophobic_ratio = sum(1 for aa in seq if aa in "AILMFWV") / length
        aromatic_ratio = sum(1 for aa in seq if aa in "FWY") / length

        return {
            "length": length,
            "molecular_weight": round(mw, 2),
            "net_charge": net_charge,
            "hydrophobicity": round(avg_hydro, 3),
            "isoelectric_point": round(pI, 2),
            "positive_residues": n_pos,
            "negative_residues": n_neg,
            "hydrophobic_ratio": round(hydrophobic_ratio, 3),
            "aromatic_ratio": round(aromatic_ratio, 3),
        }

    def _compute_pI(self, sequence: str) -> float:
        """Compute isoelectric point using bisection method."""
        seq = sequence.upper()

        def charge_at_pH(pH: float) -> float:
            charge = 0.0
            charge += 1.0 / (1.0 + 10 ** (pH - self.pka_table["N_term"]))
            charge -= 1.0 / (1.0 + 10 ** (self.pka_table["C_term"] - pH))
            for aa in seq:
                if aa in "KRH":
                    charge += 1.0 / (1.0 + 10 ** (pH - self.pka_table[aa]))
                elif aa in "DE":
                    charge -= 1.0 / (1.0 + 10 ** (self.pka_table[aa] - pH))
                elif aa == "C":
                    charge -= 1.0 / (1.0 + 10 ** (self.pka_table["C"] - pH))
                elif aa == "Y":
                    charge -= 1.0 / (1.0 + 10 ** (self.pka_table["Y"] - pH))
            return charge

        low, high = 0.0, 14.0
        for _ in range(50):
            mid = (low + high) / 2
            if charge_at_pH(mid) > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def batch_compute(self, sequences: List[str]) -> List[Dict]:
        return [self.compute_properties(seq) for seq in sequences]

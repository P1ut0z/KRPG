import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPPredictor(nn.Module):
    def __init__(self, input_dim: int = 26, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

    def predict(self, sequences: List[str], device: Optional[str] = None) -> List[Dict]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        results = []

        for seq in sequences:
            features = self._extract_features(seq)
            feat_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                score = self(feat_tensor).item()

            results.append({
                "sequence": seq,
                "amp_score": round(score, 4),
                "is_amp_candidate": score > 0.5,
            })

        return results

    def _extract_features(self, sequence: str) -> List[float]:
        seq = sequence.upper()
        length = len(seq)
        if length == 0:
            return [0.0] * 26

        aa_composition = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            aa_composition[aa] = seq.count(aa) / length

        charged = sum(seq.count(aa) for aa in "KRHDE")
        positive = sum(seq.count(aa) for aa in "KRH")
        negative = sum(seq.count(aa) for aa in "DE")
        hydrophobic = sum(seq.count(aa) for aa in "AILMFWV")
        aromatic = sum(seq.count(aa) for aa in "FWY")
        polar = sum(seq.count(aa) for aa in "NQST")
        small = sum(seq.count(aa) for aa in "GAC")

        features = [
            min(length / 50.0, 1.0),
            charged / max(length, 1),
            positive / max(length, 1),
            negative / max(length, 1),
            hydrophobic / max(length, 1),
            aromatic / max(length, 1),
            polar / max(length, 1),
            small / max(length, 1),
            (positive - negative) / max(length, 1),
        ]

        for aa in "ACDEFGHIKLMNPQRSTVWY":
            features.append(aa_composition[aa])

        return features[:26]


class RuleBasedAMPPredictor:
    def __init__(self):
        self.positive_aas = set("KRH")
        self.hydrophobic_aas = set("AILMFWV")
        self.negative_aas = set("DE")
        self.aromatic_aas = set("FWY")

    def predict(self, sequence: str) -> Dict:
        seq = sequence.upper()
        length = len(seq)
        if length < 5:
            return {"sequence": seq, "amp_score": 0.0, "is_amp_candidate": False, "reason": "Too short"}

        n_pos = sum(1 for aa in seq if aa in self.positive_aas)
        n_hydro = sum(1 for aa in seq if aa in self.hydrophobic_aas)
        n_neg = sum(1 for aa in seq if aa in self.negative_aas)
        n_aro = sum(1 for aa in seq if aa in self.aromatic_aas)

        charge = n_pos - n_neg
        hydrophobic_ratio = n_hydro / length
        positive_ratio = n_pos / length
        aromatic_ratio = n_aro / length

        score = 0.0
        reasons = []

        if 2 <= charge <= 9:
            score += 0.3
            reasons.append(f"Good net charge ({charge})")
        else:
            reasons.append(f"Suboptimal charge ({charge})")

        if 0.3 <= hydrophobic_ratio <= 0.6:
            score += 0.25
            reasons.append(f"Good hydrophobicity ({hydrophobic_ratio:.2f})")
        else:
            reasons.append(f"Suboptimal hydrophobicity ({hydrophobic_ratio:.2f})")

        if positive_ratio >= 0.15:
            score += 0.2
            reasons.append(f"Good positive residue ratio ({positive_ratio:.2f})")

        if 10 <= length <= 30:
            score += 0.15
            reasons.append(f"Good length ({length})")
        else:
            reasons.append(f"Suboptimal length ({length})")

        if aromatic_ratio >= 0.05:
            score += 0.1
            reasons.append(f"Aromatic residues present ({aromatic_ratio:.2f})")

        is_candidate = score >= 0.5

        return {
            "sequence": seq,
            "amp_score": round(score, 4),
            "is_amp_candidate": is_candidate,
            "details": {
                "length": length,
                "net_charge": charge,
                "hydrophobic_ratio": round(hydrophobic_ratio, 3),
                "positive_ratio": round(positive_ratio, 3),
                "aromatic_ratio": round(aromatic_ratio, 3),
            },
            "reasons": reasons,
        }


class CompositeScorer:
    """Weighted composite scoring across all validation dimensions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "amp": 0.35,
            "toxicity_inv": 0.25,
            "stability": 0.20,
            "novelty": 0.20,
        }

    def score(self, amp_score: float, toxicity_score: float,
              stability_score: float, is_novel: bool) -> Dict:
        novelty_val = 1.0 if is_novel else 0.5
        composite = (
            amp_score * self.weights["amp"]
            + (1.0 - toxicity_score) * self.weights["toxicity_inv"]
            + stability_score * self.weights["stability"]
            + novelty_val * self.weights["novelty"]
        )
        return {
            "composite_score": round(composite, 4),
            "breakdown": {
                "amp_contribution": round(amp_score * self.weights["amp"], 4),
                "toxicity_contribution": round((1.0 - toxicity_score) * self.weights["toxicity_inv"], 4),
                "stability_contribution": round(stability_score * self.weights["stability"], 4),
                "novelty_contribution": round(novelty_val * self.weights["novelty"], 4),
            },
        }

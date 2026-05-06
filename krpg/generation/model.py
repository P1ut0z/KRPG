import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, 1024, 1024)).view(1, 1, 1024, 1024) == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = self.causal_mask[:, :, :T, :T]
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        y = self.proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PromptEncoder(nn.Module):
    """Encodes structured prompt properties into a conditioning vector for the generator."""

    FEATURE_NAMES = [
        "length_mid_norm",
        "length_span_norm",
        "charge_mid_norm",
        "charge_span_norm",
        "hydrophobic_mid",
        "hydrophobic_span",
        "positive_ratio",
        "aromatic_ratio",
        "target_broad",
        "target_gram_negative",
        "target_gram_positive",
        "constraint_low_toxicity",
        "constraint_stability",
        "motif_amphipathic",
        "evidence_confidence",
        "novelty_threshold",
    ]
    PROPERTY_DIM = len(FEATURE_NAMES)

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Sequential(
            nn.Linear(self.PROPERTY_DIM, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        import re
        return [float(n.replace("+", "")) for n in re.findall(r"\+?\d+(?:\.\d+)?", text or "")]

    @classmethod
    def _range_from_text(cls, text: str, fallback: Optional[List[float]] = None) -> Optional[List[float]]:
        nums = cls._extract_numbers(text)
        if len(nums) >= 2:
            return [nums[0], nums[1]]
        if len(nums) == 1:
            return [nums[0], nums[0]]
        return fallback

    @staticmethod
    def _context_feature_range(context: Dict, feature: str) -> Optional[List[float]]:
        for item in context.get("property_constraints", []):
            if isinstance(item, dict) and item.get("feature") == feature and "range" in item:
                values = item["range"]
                if isinstance(values, (list, tuple)) and len(values) >= 2:
                    return [float(values[0]), float(values[1])]
        return None

    @staticmethod
    def _has_context_text(context: Dict, keys: List[str], terms: List[str]) -> bool:
        haystack = []
        for key in keys:
            for item in context.get(key, []):
                if isinstance(item, dict):
                    haystack.append(str(item.get("text", "")))
                    haystack.append(str(item.get("properties", "")))
                else:
                    haystack.append(str(item))
        text = " ".join(haystack).lower()
        return any(term.lower() in text for term in terms)

    def encode_spec(self, design_spec: Dict) -> torch.Tensor:
        """Convert a design spec dict into a fixed-size feature vector."""
        length_str = design_spec.get("preference", "")
        target = design_spec.get("target", "").lower()
        activity = design_spec.get("activity", "").lower()
        constraint = design_spec.get("constraint", "").lower()
        context = design_spec.get("retrieved_context") or design_spec.get("knowledge") or {}

        length_range = design_spec.get("length_range")
        if not length_range:
            length_range = self._context_feature_range(context, "length")
        if not length_range:
            length_range = self._range_from_text(length_str, [12.0, 30.0])
        length_mid = (float(length_range[0]) + float(length_range[1])) / 2.0
        length_span = abs(float(length_range[1]) - float(length_range[0]))

        charge_range = design_spec.get("charge_range")
        if not charge_range:
            charge_range = self._context_feature_range(context, "net_charge")
        if not charge_range:
            charge_range = self._range_from_text(design_spec.get("charge", "") or length_str, [2.0, 9.0])
        charge_mid = (float(charge_range[0]) + float(charge_range[1])) / 2.0
        charge_span = abs(float(charge_range[1]) - float(charge_range[0]))

        hydro_range = design_spec.get("hydrophobicity_range")
        if not hydro_range:
            hydro_range = self._context_feature_range(context, "hydrophobic_ratio")
        if not hydro_range:
            hydro_range = [0.35, 0.60]
        hydrophobic_mid = (float(hydro_range[0]) + float(hydro_range[1])) / 2.0
        hydrophobic_span = abs(float(hydro_range[1]) - float(hydro_range[0]))

        target_text = f"{target} {activity}"
        target_broad = 1.0 if "broad" in target_text or "广谱" in target_text else 0.0
        target_gram_neg = 1.0 if "gram-negative" in target_text or "革兰氏阴性" in target_text else 0.0
        target_gram_pos = 1.0 if "gram-positive" in target_text or "革兰氏阳性" in target_text else 0.0

        constraint_low_tox = 1.0 if "low toxicity" in constraint or "低毒" in constraint else 0.0
        constraint_stability = 1.0 if "stability" in constraint or "稳定" in constraint else 0.0
        motif_amphipathic = 1.0 if (
            "amphipathic" in length_str.lower()
            or "两亲" in length_str
            or self._has_context_text(context, ["motif_hints", "design_rules"], ["amphipathic", "两亲"])
        ) else 0.0

        evidence_confidence = 0.0
        evidence_count = 0
        for items in context.values() if isinstance(context, dict) else []:
            if isinstance(items, list):
                evidence_count += len(items)
                for item in items:
                    if isinstance(item, dict):
                        evidence_confidence += float(item.get("relevance", item.get("confidence", 0.5)))
                    else:
                        evidence_confidence += 0.5
        if evidence_count > 0:
            evidence_confidence = min(evidence_confidence / evidence_count, 1.0)

        novelty_threshold = float(design_spec.get("novelty_threshold", 0.8))

        features = torch.tensor([[
            min(length_mid / 50.0, 1.0),
            min(length_span / 50.0, 1.0),
            max(min(charge_mid / 10.0, 1.0), -1.0),
            min(charge_span / 10.0, 1.0),
            max(min(hydrophobic_mid, 1.0), 0.0),
            max(min(hydrophobic_span, 1.0), 0.0),
            0.25,
            0.1,
            target_broad, target_gram_neg, target_gram_pos,
            constraint_low_tox, constraint_stability,
            motif_amphipathic,
            evidence_confidence,
            novelty_threshold,
        ]], dtype=torch.float32)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)


class KRPGGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 512, max_seq_len: int = 64,
                 dropout: float = 0.1, pad_token_id: int = 0,
                 use_prompt_conditioning: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.use_prompt_conditioning = use_prompt_conditioning

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        if use_prompt_conditioning:
            self.prompt_encoder = PromptEncoder(d_model)
            self.prompt_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                prompt_vector: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb

        if prompt_vector is not None and self.use_prompt_conditioning:
            x = x + self.prompt_scale * prompt_vector.unsqueeze(1)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    def encode_prompt(self, design_spec: Dict) -> torch.Tensor:
        """Encode a design specification into a conditioning vector."""
        if not self.use_prompt_conditioning:
            return None
        features = self.prompt_encoder.encode_spec(design_spec)
        device = next(self.prompt_encoder.parameters()).device
        features = features.to(device)
        return self.prompt_encoder(features)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None, eos_token_id: Optional[int] = None,
                 prompt_vector: Optional[torch.Tensor] = None,
                 min_new_tokens: int = 0,
                 forbidden_token_ids: Optional[List[int]] = None) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        initial_len = input_ids.shape[1]
        forbidden_token_ids = forbidden_token_ids or []

        for _ in range(max_new_tokens):
            seq_len = input_ids.shape[1]
            if seq_len > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            with torch.no_grad():
                logits, _ = self.forward(input_ids, prompt_vector=prompt_vector)
                next_logits = logits[:, -1, :]

            generated_steps = max(input_ids.shape[1] - initial_len, 0)
            for token_id in forbidden_token_ids:
                if 0 <= token_id < next_logits.shape[-1] and token_id != eos_token_id:
                    next_logits[:, token_id] = float("-inf")
            if eos_token_id is not None and generated_steps < min_new_tokens:
                next_logits[:, eos_token_id] = float("-inf")

            if temperature > 0:
                next_logits = next_logits / temperature

            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = values[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return input_ids

    def get_model_size(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_millions": total_params / 1_000_000,
        }

import json
import os
import re
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from krpg.generation.tokenizer import AminoAcidTokenizer
from krpg.generation.model import KRPGGenerator


class AMPSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: AminoAcidTokenizer, max_length: int = 50):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = self.tokenizer.encode(seq, add_special_tokens=True, max_length=self.max_length)
        return torch.tensor(token_ids, dtype=torch.long)


class SequenceGenerator:
    def __init__(self, model: KRPGGenerator, tokenizer: AminoAcidTokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def _infer_min_new_tokens(design_spec: Optional[Dict], max_length: int) -> int:
        if not design_spec:
            return 0
        length_range = design_spec.get("length_range")
        if not length_range:
            preference = design_spec.get("preference", "")
            nums = [int(float(n.replace("+", ""))) for n in re.findall(r"\+?\d+(?:\.\d+)?", preference)]
            if len(nums) >= 2 and "length" in preference.lower():
                length_range = [nums[0], nums[1]]
        if not length_range:
            return 0
        return max(0, min(int(length_range[0]), max_length))

    def _forbidden_generation_tokens(self) -> List[int]:
        forbidden = [
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.unk_token_id,
        ]
        mask_id = self.tokenizer.vocab.get("<MASK>")
        if mask_id is not None:
            forbidden.append(mask_id)
        return forbidden

    def train_model(self, dataset: AMPSequenceDataset, n_epochs: int = 10, batch_size: int = 16,
                    learning_rate: float = 1e-3, save_path: Optional[str] = None,
                    design_spec: Optional[Dict] = None) -> Dict[str, List[float]]:
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        history = {"loss": []}

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                pv = None
                if design_spec and self.model.use_prompt_conditioning:
                    pv = self.model.encode_prompt(design_spec).to(self.device)
                    pv = pv.expand(batch.shape[0], -1)

                _, loss = self.model(input_ids=batch, labels=batch, prompt_vector=pv)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, save_path)

        return history

    def generate_sequences(self, prompt_ids: Optional[torch.Tensor] = None,
                           n_sequences: int = 10, max_length: int = 50,
                           temperature: float = 1.0, top_k: Optional[int] = 40,
                           top_p: Optional[float] = 0.9,
                           design_spec: Optional[Dict] = None,
                           prompt_record: Optional[Dict] = None,
                           round_num: int = 1) -> List[Dict]:
        if prompt_record:
            record_spec = dict(prompt_record.get("design_spec", {}))
            if prompt_record.get("retrieved_context"):
                record_spec["retrieved_context"] = prompt_record["retrieved_context"]
            design_spec = design_spec or record_spec

        if prompt_ids is None:
            prompt_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

        prompt_ids = prompt_ids.to(self.device)

        prompt_vector = None
        if design_spec and self.model.use_prompt_conditioning:
            prompt_vector = self.model.encode_prompt(design_spec).to(self.device)
        min_new_tokens = self._infer_min_new_tokens(design_spec, max_length)
        forbidden_token_ids = self._forbidden_generation_tokens()

        results = []

        for i in range(n_sequences):
            generated = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                prompt_vector=prompt_vector,
                min_new_tokens=min_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
            )
            seq_ids = generated[0].tolist()
            sequence = self.tokenizer.decode(seq_ids)

            with torch.no_grad():
                logits, _ = self.model(generated, prompt_vector=prompt_vector)
                probs = F.softmax(logits, dim=-1)
                token_probs = [probs[0, j, tid].item() for j, tid in enumerate(seq_ids) if tid != self.tokenizer.pad_token_id]
                avg_prob = sum(token_probs) / max(len(token_probs), 1)

            result = {
                "sequence": sequence,
                "token_ids": seq_ids,
                "avg_log_prob": avg_prob,
                "avg_token_prob": avg_prob,
                "length": len(sequence),
                "round": round_num,
            }
            if design_spec:
                result["design_spec"] = design_spec
            if prompt_record:
                result["prompt"] = prompt_record.get("rendered_prompt")
                result["prompt_record"] = prompt_record
                result["retrieved_knowledge"] = prompt_record.get("retrieved_context")
            results.append(result)

        return results

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_state = checkpoint["model_state_dict"]
        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model_state and getattr(value, "shape", None) == model_state[key].shape
        }
        skipped = sorted(set(checkpoint_state) - set(compatible_state))
        self.model.load_state_dict(compatible_state, strict=False)
        if skipped:
            checkpoint["skipped_state_keys"] = skipped
        return checkpoint.get("history", None)

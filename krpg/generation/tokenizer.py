from typing import List, Optional


class AminoAcidTokenizer:
    STANDARD_AMINO_ACIDS = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]

    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"]

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self._build_vocab()

    def _build_vocab(self):
        idx = 0
        for token in self.SPECIAL_TOKENS:
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            idx += 1
        for aa in self.STANDARD_AMINO_ACIDS:
            self.vocab[aa] = idx
            self.inverse_vocab[idx] = aa
            idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<UNK>"]

    def encode(self, sequence: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for char in sequence.upper():
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        if max_length is not None and len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        return tokens[:max_length] if max_length else tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, "<UNK>")
            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars)

    def batch_encode(self, sequences: List[str], max_length: int = 50) -> List[List[int]]:
        return [self.encode(seq, max_length=max_length) for seq in sequences]

    def batch_decode(self, batch_ids: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]

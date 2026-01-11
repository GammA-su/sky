"""Data schemas and tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


class ByteTokenizer:
    """Deterministic byte-level tokenizer with fixed vocab."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    offset = 3
    vocab_size = 256 + offset

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        data = text.encode("utf-8", errors="replace")
        ids = [b + self.offset for b in data]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_out = bytearray()
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            byte = idx - self.offset
            if 0 <= byte <= 255:
                bytes_out.append(byte)
        return bytes_out.decode("utf-8", errors="replace")


@dataclass
class SynthExample:
    prompt: str
    completion: str
    state_ids: List[int]
    hypotheses: List[str] | None = None
    correct_mask: List[bool] | None = None

"""Generic UTF-8 completion guard for byte-level (Tekken) decoding.

The decoder emits one token per frame, and a multi-byte character can be split into a lead-byte
token plus continuation-byte tokens. Greedy decoding could strand a lead byte (it decodes to
U+FFFD). The guard tracks the incomplete tail and masks, through an additive logit bias, every
token that would make the byte stream invalid UTF-8. It restricts nothing else: all scripts stay
reachable.
"""
from __future__ import annotations

import numpy as np
import torch

NEG = float("-inf")


def incomplete_tail(b: bytes) -> bytes:
    """Trailing bytes of a valid UTF-8 prefix that form an unfinished character (else ``b""``)."""
    n = len(b)
    for k in range(1, min(4, n) + 1):
        lead = b[n - k]
        if lead < 0x80 or lead >= 0xC0:
            need = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
            return b[n - k:] if k < need else b""
    return b""


def valid_prefix(cand: bytes) -> bool:
    """True if ``cand`` is complete UTF-8 optionally followed by one valid unfinished character."""
    i, n = 0, len(cand)
    while i < n:
        b0 = cand[i]
        if 0x80 <= b0 < 0xC0 or b0 >= 0xF8:
            return False
        length = 1 if b0 < 0x80 else 2 if b0 < 0xE0 else 3 if b0 < 0xF0 else 4
        end = i + length
        for j in range(i + 1, min(end, n)):
            if not 0x80 <= cand[j] < 0xC0:
                return False
        if end > n:
            # unfinished: make sure the prefix can still become a valid character
            return _can_complete(cand[i:])
        try:
            cand[i:end].decode("utf-8")
        except UnicodeDecodeError:
            return False
        i = end
    return True


def _can_complete(part: bytes) -> bool:
    for tail in (b"\x80" * 3, b"\xbf" * 3, b"\x80\xbf\x80", b"\x9f\xbf\xbf", b"\xa0\x80\x80"):
        for k in range(1, 4):
            try:
                (part + tail[:k]).decode("utf-8")
                return True
            except UnicodeDecodeError:
                continue
    return False


class Utf8Guard:
    def __init__(self, token_bytes: list[bytes], device, boundary: np.ndarray | None = None):
        self.tb = token_bytes
        self.device = device
        V = len(token_bytes)
        self._cont = [i for i, bp in enumerate(token_bytes) if bp and 0x80 <= bp[0] < 0xC0]
        if boundary is None:
            boundary = np.zeros(V, dtype=np.float32)
            for i, bp in enumerate(token_bytes):
                if bp and not bp.isascii() and not valid_prefix(bp):
                    boundary[i] = NEG
            boundary[self._cont] = NEG         # cannot start a character
        self.boundary_np = boundary
        self._boundary = torch.from_numpy(boundary).to(device)
        self._cache: dict[bytes, torch.Tensor] = {}

    def bias_for(self, pending: bytes) -> torch.Tensor:
        if not pending:
            return self._boundary
        b = self._cache.get(pending)
        if b is None:
            arr = np.full(len(self.tb), NEG, dtype=np.float32)
            for i in self._cont:
                if valid_prefix(pending + self.tb[i]):
                    arr[i] = 0.0
            if not np.isfinite(arr).any():     # degenerate state: let any continuation through
                arr[self._cont] = 0.0
            b = self._cache[pending] = torch.from_numpy(arr).to(self.device)
        return b

    def advance(self, pending: bytes, token: int) -> bytes:
        return incomplete_tail(pending + self.tb[token])

    def strip_incomplete(self, ids: list[int]) -> list[int]:
        """Drop trailing tokens that leave an unfinished character."""
        pending, last = b"", 0
        for i, t in enumerate(ids):
            pending = incomplete_tail(pending + self.tb[t])
            if not pending:
                last = i + 1
        return ids[:last]

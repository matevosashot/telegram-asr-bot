"""Greedy frame-synchronous decoding: one token per 80 ms frame."""
from __future__ import annotations

import torch

from .constants import LEFT_PAD_FRAMES, TOKEN_BOS, TOKEN_EOS, TOKEN_PAD


def prompt_len(delay: int) -> int:
    return 1 + LEFT_PAD_FRAMES + delay


@torch.no_grad()
def greedy_decode(model, emb: torch.Tensor, delay: int) -> list[int]:
    """``emb`` [A, H]: all decoder-frame embeddings of the padded audio. Returns the token ids."""
    dev, guard = model.device, model.guard
    P, A = prompt_len(delay), emb.shape[0]
    size = min(model.cfg.dec_window, -(-(A + 16) // 512) * 512)
    cache = model.acquire_cache(size)
    try:
        ada = model.net.decoder.ada_scales(delay, model.dtype)
        toks = torch.tensor([[TOKEN_BOS] + [TOKEN_PAD] * (P - 1)], device=dev)
        logits = model.decode_rows(emb[:P].unsqueeze(0), toks, ada, cache)   # prefill + first token
        cache.set_pos(P)
        out: list[int] = []
        pending, last = b"", TOKEN_PAD
        tok_t = torch.zeros(1, 1, dtype=torch.long, device=dev)
        for f in range(P - 1, A):
            if f >= P:
                tok_t.fill_(last)
                logits = model.decode_rows(emb[f].view(1, 1, -1), tok_t, ada, cache)
                cache.advance()
            t = int((logits.float() + guard.bias_for(pending)).argmax(-1).item())
            out.append(t)
            pending = guard.advance(pending, t)
            last = TOKEN_PAD if t == TOKEN_EOS else t
        return out
    finally:
        model.release_cache(cache)

"""Pure-torch Voxtral-Realtime encoder / projector / decoder.

Weights are plain parameters filled with ``load_state_dict(assign=True)`` so
building the net costs nothing; there is no transformers dependency.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # encoder
    enc_layers: int = 32
    enc_dim: int = 1280
    enc_heads: int = 32
    enc_head_dim: int = 64
    enc_ffn: int = 5120
    enc_window: int = 750
    enc_theta: float = 1_000_000.0
    # decoder
    dec_layers: int = 26
    dec_dim: int = 3072
    dec_heads: int = 32
    dec_kv_heads: int = 8
    dec_head_dim: int = 128
    dec_ffn: int = 9216
    dec_window: int = 8192
    dec_theta: float = 1_000_000.0
    vocab_size: int = 131072
    norm_eps: float = 1e-5
    ada_dim: int = 32          # width of the delay-conditioning bottleneck
    time_theta: float = 10000.0

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=1)


def rms_norm(x, weight, eps):
    """HF Voxtral RMSNorm: fp32 normalise, cast back, then scale in the model dtype."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def _rot_half(t):
    h = t.shape[-1] // 2
    return torch.cat((-t[..., h:], t[..., :h]), dim=-1)


@torch.library.custom_op("asr_bot_voxtral::kv_write", mutates_args=("k", "v"))
def kv_write(k: torch.Tensor, v: torch.Tensor, kk: torch.Tensor, vv: torch.Tensor,
             slots: torch.Tensor, layer: int) -> None:
    """Write new K/V rows into the ring cache. A custom op so torch.compile mutates the
    cache in place instead of functionalising (copying) the whole cache every step."""
    k[layer].index_copy_(2, slots, kk)
    v[layer].index_copy_(2, slots, vv)


@torch.library.custom_op("asr_bot_voxtral::kpos_write", mutates_args=("kpos",))
def kpos_write(kpos: torch.Tensor, slots: torch.Tensor, qpos: torch.Tensor) -> None:
    kpos.index_copy_(0, slots, qpos)


def _P(*shape):
    return nn.Parameter(torch.empty(*shape), requires_grad=False)


# ---------------------------------------------------------------- encoder

class EncLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d, qd, f = c.enc_dim, c.enc_heads * c.enc_head_dim, c.enc_ffn
        self.attn_norm = _P(d)
        self.wqkv, self.bqkv = _P(3 * qd, d), _P(3 * qd)
        self.wo, self.bo = _P(d, qd), _P(d)
        self.ffn_norm = _P(d)
        self.w13 = _P(2 * f, d)
        self.w2, self.b2 = _P(d, f), _P(d)


class Encoder(nn.Module):
    """Causal sliding-window Whisper-style encoder + 4x frame grouping + projector."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        d = c.enc_dim
        self.conv1_w, self.conv1_b = _P(d, 128, 3), _P(d)
        self.conv2_w, self.conv2_b = _P(d, d, 3), _P(d)
        self.layers = nn.ModuleList(EncLayer(c) for _ in range(c.enc_layers))
        self.norm = _P(d)
        self.proj1, self.proj2 = _P(c.dec_dim, d * 4), _P(c.dec_dim, c.dec_dim)
        hd = c.enc_head_dim
        self.register_buffer("inv_freq", 1.0 / (c.enc_theta ** (
            torch.arange(0, hd, 2, dtype=torch.float) / hd)), persistent=False)

    def init_buffers(self, device):
        hd = self.c.enc_head_dim
        self.inv_freq = 1.0 / (self.c.enc_theta ** (
            torch.arange(0, hd, 2, dtype=torch.float, device=device) / hd))

    def new_state(self):
        return {"kv": [None] * self.c.enc_layers, "pos": 0, "rest": None}

    @torch.no_grad()
    def forward_chunk(self, x, state):
        """x: conv-stem tokens [n, 1280] -> projected decoder-frame embeddings [m, 3072] or None."""
        c = self.c
        n = x.shape[0]
        H, hd = c.enc_heads, c.enc_head_dim
        pos0 = state["pos"]
        x = x.unsqueeze(0)
        pos = torch.arange(pos0, pos0 + n, device=x.device)
        freqs = pos[:, None].float() * self.inv_freq.to(x.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)[None, None]
        sin = emb.sin().to(x.dtype)[None, None]
        keep = c.enc_window - 1
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, c.norm_eps)
            qkv = F.linear(h, L.wqkv, L.bqkv)
            q, k, v = (t.view(1, n, H, hd).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            old = state["kv"][li]
            if old is None:
                kf, vf, S = k, v, 0
            else:
                kf, vf, S = torch.cat([old[0], k], 2), torch.cat([old[1], v], 2), old[0].shape[2]
            state["kv"][li] = (kf[:, :, -keep:], vf[:, :, -keep:])
            rel = (torch.arange(S + n, device=x.device)[None, :]
                   - torch.arange(S, S + n, device=x.device)[:, None])
            mask = ((rel <= 0) & (rel > -c.enc_window))[None, None]
            o = F.scaled_dot_product_attention(q, kf, vf, attn_mask=mask, scale=hd ** -0.5)
            x = x + F.linear(o.transpose(1, 2).reshape(1, n, H * hd), L.wo, L.bo)
            h = rms_norm(x, L.ffn_norm, c.norm_eps)
            gu = F.linear(h, L.w13)
            x = x + F.linear(F.silu(gu[..., :c.enc_ffn]) * gu[..., c.enc_ffn:], L.w2, L.b2)
        state["pos"] = pos0 + n
        x = rms_norm(x, self.norm, c.norm_eps)[0]
        if state["rest"] is not None:
            x = torch.cat([state["rest"], x], 0)
        m = x.shape[0] // 4
        state["rest"] = x[m * 4:] if m * 4 < x.shape[0] else None
        if m == 0:
            return None
        g = x[:m * 4].reshape(m, -1)
        return F.linear(F.gelu(F.linear(g, self.proj1)), self.proj2)


# ---------------------------------------------------------------- decoder

class DecLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.dec_dim
        qd, kd = c.dec_heads * c.dec_head_dim, c.dec_kv_heads * c.dec_head_dim
        self.attn_norm = _P(d)
        self.wqkv = _P(qd + 2 * kd, d)
        self.wo = _P(d, qd)
        self.ffn_norm = _P(d)
        self.w13 = _P(2 * c.dec_ffn, d)
        self.w2 = _P(d, c.dec_ffn)
        # delay-conditioned RMSNorm: scale = up(gelu(down(time_embedding(delay))))
        self.ada_down = _P(c.ada_dim, d)
        self.ada_up = _P(d, c.ada_dim)


SLACK = 512   # unused slots after the ring so the attention span never equals the dim size


class DecoderCache:
    """Static ring KV cache for ``batch`` hypotheses and ``size`` positions."""

    def __init__(self, c: Config, batch: int, size: int, device, dtype):
        shape = (c.dec_layers, batch, c.dec_kv_heads, size + SLACK, c.dec_head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.kpos = torch.full((size + SLACK,), -1, device=device, dtype=torch.long)
        self.pos = torch.zeros(1, device=device, dtype=torch.long)   # next row position
        self.size, self.batch = size, batch
        self.hpos = 0                                                # host copy of ``pos``
        self.runners: dict = {}                                      # span -> CUDA-graph step

    def reset(self):
        self.kpos.fill_(-1)
        self.set_pos(0)

    def set_pos(self, p: int):
        self.pos.fill_(p)
        self.hpos = p

    def advance(self, n: int = 1):
        self.pos += n
        self.hpos += n

    def span(self, n: int, bucket: int = 512) -> int:
        """Attention span (slots) for ``n`` rows at the current position."""
        return min(self.size, -(-(self.hpos + n) // bucket) * bucket)

    def reorder(self, idx: torch.Tensor):
        """Permute the hypotheses in place (CUDA graphs hold these buffers' addresses);
        only the slots that can hold keys are moved."""
        n = self.span(0)
        for t in (self.k, self.v):
            view = t[:, :, :, :n]
            view.copy_(view.index_select(1, idx))


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = _P(c.vocab_size, c.dec_dim)
        self.norm = _P(c.dec_dim)
        self.layers = nn.ModuleList(DecLayer(c) for _ in range(c.dec_layers))
        hd = c.dec_head_dim
        self.register_buffer("inv_freq", 1.0 / (c.dec_theta ** (
            torch.arange(0, hd, 2, dtype=torch.float) / hd)), persistent=False)

    def init_buffers(self, device):
        hd = self.c.dec_head_dim
        self.inv_freq = 1.0 / (self.c.dec_theta ** (
            torch.arange(0, hd, 2, dtype=torch.float, device=device) / hd))

    def ada_scales(self, delay: int, dtype=None) -> torch.Tensor:
        """[layers, dim] FFN-norm scales for a transcription delay of ``delay`` frames.

        The delay is embedded sinusoidally (computed in the model dtype, as the reference does)."""
        c = self.c
        dtype = dtype or self.embed.dtype
        dev = self.embed.device
        half = c.dec_dim // 2
        inv = torch.exp(-math.log(c.time_theta) * torch.arange(half, device=dev).float() / half)
        t = torch.full((1,), float(delay), device=dev, dtype=dtype)
        emb = t * inv.to(dtype)
        cond = torch.cat((emb.cos(), emb.sin()))
        return torch.stack([F.linear(F.gelu(F.linear(cond, L.ada_down)), L.ada_up)
                            for L in self.layers])

    def forward(self, audio, tokens, pos0, ada, k, v, kpos, span: int = 0):
        """Decode ``n`` consecutive frames for ``B`` hypotheses; returns logits [B, V] for the last row.

        audio  [B, n, H] projected audio embeddings; tokens [B, n] previous-token ids;
        pos0   long tensor [1]: absolute position of row 0; ada [layers, H];
        k, v   [layers, B, Hkv, T, D] ring caches (updated in place); kpos [T] long;
        span   attention span in slots (static; >= pos0 + n unless the ring has wrapped).
        """
        c = self.c
        B, n, _ = audio.shape
        Hh, Hk, hd = c.dec_heads, c.dec_kv_heads, c.dec_head_dim
        T = kpos.shape[0] - SLACK
        span = span or T
        x = audio + self.embed[tokens]
        qpos = pos0 + torch.arange(n, device=audio.device)                 # [n]
        slots = qpos % T
        kpos_write(kpos, slots, qpos)
        kp = kpos[:span]
        valid = ((kp[None, :] >= 0) & (kp[None, :] <= qpos[:, None])
                 & (qpos[:, None] - kp[None, :] < c.dec_window))            # [n, span]
        mask = valid[None, None]
        freqs = qpos[:, None].float() * self.inv_freq
        cos = freqs.cos().to(x.dtype)[None, None]                           # [1,1,n,hd/2]
        sin = freqs.sin().to(x.dtype)[None, None]

        def rope(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin], dim=-1)

        qd, kd = Hh * hd, Hk * hd
        for li, L in enumerate(self.layers):
            h = rms_norm(x, L.attn_norm, c.norm_eps)
            qkv = F.linear(h, L.wqkv)
            q = rope(qkv[..., :qd].view(B, n, Hh, hd).transpose(1, 2))
            kk = rope(qkv[..., qd:qd + kd].view(B, n, Hk, hd).transpose(1, 2))
            vv = qkv[..., qd + kd:].view(B, n, Hk, hd).transpose(1, 2)
            kv_write(k, v, kk, vv, slots, li)
            if n == 1:   # decode: the 4 query heads of a kv group are 4 rows of one kv head
                sc = torch.matmul(q.reshape(B, Hk, Hh // Hk, hd),
                                  k[li][:, :, :span].transpose(-1, -2)).float() * hd ** -0.5
                pr = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1).to(x.dtype)
                o = torch.matmul(pr, v[li][:, :, :span]).reshape(B, 1, qd)
            else:
                o = F.scaled_dot_product_attention(
                    q, k[li][:, :, :span], v[li][:, :, :span], attn_mask=mask,
                    scale=hd ** -0.5, enable_gqa=True).transpose(1, 2).reshape(B, n, qd)
            x = x + F.linear(o, L.wo)
            h = rms_norm(x, L.ffn_norm, c.norm_eps) * (1 + ada[li])
            gu = F.linear(h, L.w13)
            x = x + F.linear(F.silu(gu[..., :c.dec_ffn]) * gu[..., c.dec_ffn:], L.w2)
        return F.linear(rms_norm(x[:, -1], self.norm, c.norm_eps), self.embed)  # [B, V]


class ASRNet(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)

    def init_buffers(self, device):
        self.encoder.init_buffers(device)
        self.decoder.init_buffers(device)

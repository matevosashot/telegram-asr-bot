"""Voxtral: weight lifecycle (load / offload), compile, and offline transcription."""
from __future__ import annotations

import json
import os
import struct
import time

import numpy as np
import torch
from safetensors import safe_open

from . import cache as _cache
from .frontend import ConvStem, MelStream, load_audio, mel_filters, to_float32
from .constants import (DEFAULT_DELAY_MS, FRAME_MS, LEFT_PAD_FRAMES, MAX_DELAY_FRAMES,
                        MIN_DELAY_FRAMES, NON_TEXT_TOKENS, RIGHT_PAD_BUFFER_FRAMES,
                        SAMPLE_RATE, SAMPLES_PER_FRAME)
from .decode import greedy_decode
from .guard import Utf8Guard
from .loader import load_flat
from .net import ASRNet, Config, DecoderCache

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_TABLES = ("tokenizer.blob", "tokenizer.offs", "guard.boundary")


class Frontend:
    """waveform -> mel -> conv stem -> encoder -> projected decoder-frame embeddings (chunkable)."""

    def __init__(self, model: "Voxtral"):
        e = model.net.encoder
        self.m = model
        self.mel = MelStream(model.mel_filters, model.device)
        self.stem = ConvStem(e.conv1_w, e.conv1_b, e.conv2_w, e.conv2_b)
        self.state = e.new_state()

    @torch.no_grad()
    def feed(self, audio: np.ndarray):
        mel = self.mel.feed(torch.from_numpy(audio).to(self.m.device))
        if mel is None:
            return None
        tok = self.stem.feed(mel.to(self.m.dtype))
        return None if tok is None else self.m.net.encoder.forward_chunk(tok, self.state)


class Voxtral:
    """Offline transcriber for a local HF-format Voxtral Mini 4B Realtime checkpoint directory.

    The checkpoint (bf16 or fp32) is converted once per requested ``dtype`` and cached in
    ``~/.cache/asr-bot/voxtral``; later loads use the cached file (see ``cache.py``).
    """

    def __init__(self, path: str, device: str = "auto", dtype: str = "auto", load: bool = True,
                 cache_dir: str | None = None, tekken: str | None = None):
        self.path = path
        self.cache_root = cache_dir
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype == "auto":
            dtype = "bf16" if self.device.type == "cuda" else "fp32"
        self.dtype_name, self.dtype = dtype, _DTYPES[dtype]
        self.file = _cache.resolve(path, dtype, cache_dir, tekken)
        with safe_open(self.file, "pt", device="cpu") as f:
            self.cfg = Config(**json.loads(f.metadata()["config"]))
            blob, offs, bound = (f.get_tensor(k) for k in _TABLES)
        blob, offs = blob.numpy().tobytes(), offs.numpy()
        tb = [blob[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        self.guard = Utf8Guard(tb, self.device, boundary=bound.numpy())
        self.token_bytes = tb
        self.default_delay_ms = DEFAULT_DELAY_MS
        with torch.device("meta"):
            self.net = ASRNet(self.cfg)
        self.net.eval()
        self.mel_filters = mel_filters().to(self.device)
        self._step = None
        self._pool: dict = {}
        self.use_graphs = self.device.type == "cuda"
        self.loaded = False
        if load:
            self.load()

    # ---- weights ------------------------------------------------------

    def load(self):
        """Materialise the weights on the device (pinned, double-buffered upload)."""
        if self.loaded:
            return
        t0 = time.time()
        sd = load_flat(self.file, self.device)
        for k in _TABLES:
            sd.pop(k)
        self.net.load_state_dict(sd, assign=True, strict=True)
        self.net.init_buffers(self.device)
        self.loaded = True
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.load_seconds = time.time() - t0

    def offload(self):
        """Free the weights (and VRAM). ``load()`` restores them; compiled code is kept."""
        if not self.loaded:
            return
        self._pool.clear()
        self.net.to("meta")
        self.loaded = False
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- decoder step -------------------------------------------------

    def acquire_cache(self, size: int) -> DecoderCache:
        size = min(size, self.cfg.dec_window)
        free = self._pool.get(size)
        cache = free.pop() if free else DecoderCache(self.cfg, 1, size, self.device, self.dtype)
        cache.reset()
        return cache

    def release_cache(self, cache: DecoderCache):
        free = self._pool.setdefault(cache.size, [])
        if cache not in free:
            free.append(cache)
        while sum(len(v) for v in self._pool.values()) > 4:
            k = next(k for k, v in self._pool.items() if v)
            self._pool[k].pop(0)

    @torch.no_grad()
    def decode_rows(self, audio, tokens, ada, cache: DecoderCache):
        """Run ``n`` frames; returns last-row logits [1, V]. Single rows replay a CUDA graph on CUDA."""
        n = audio.shape[1]
        span = cache.span(n)
        if n == 1 and self.use_graphs:
            r = cache.runners.get(span)
            if r is None:
                r = cache.runners[span] = _GraphStep(self, cache, ada, span)
            return r.run(audio, tokens, ada)
        fn = self._step if (n == 1 and self._step is not None) else self.net.decoder
        return fn(audio, tokens, cache.pos, ada, cache.k, cache.v, cache.kpos, span)

    def compile(self, cache_dir: str | None = None, warmup: bool = True):
        """``torch.compile`` the single-row decoder step (GPU) and warm it up.

        The Inductor cache (default ``<cache dir>/inductor``) makes later processes start warm."""
        import torch._dynamo
        self.load()
        d = cache_dir or os.path.join(_cache.cache_dir(self.cache_root), "inductor")
        os.makedirs(d, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(d)
        torch._dynamo.config.cache_size_limit = 64
        self._step = torch.compile(self.net.decoder, fullgraph=True, dynamic=True)
        if warmup:
            self.warmup()
        return self

    @torch.no_grad()
    def warmup(self, size: int = 1024, delay: int = 6):
        cache = self.acquire_cache(size)
        ada = self.net.decoder.ada_scales(delay, self.dtype)
        a = torch.zeros(1, 1, self.cfg.dec_dim, device=self.device, dtype=self.dtype)
        t = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.decode_rows(a, t, ada, cache)
        self.release_cache(cache)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ---- transcription -------------------------------------------------

    def delay_frames(self, delay_ms: float | None) -> int:
        ms = self.default_delay_ms if delay_ms is None else delay_ms
        d = int(round(ms / FRAME_MS))
        if not MIN_DELAY_FRAMES <= d <= MAX_DELAY_FRAMES:
            raise ValueError(f"delay_ms must be within {MIN_DELAY_FRAMES * FRAME_MS}..{MAX_DELAY_FRAMES * FRAME_MS}")
        return d

    def encode(self, wav: np.ndarray, delay: int, chunk_s: float = 30.0) -> torch.Tensor:
        """Pad like the reference (left 32 frames; right: align to a frame + delay + 1 + 10 frames) and encode."""
        pad_l = LEFT_PAD_FRAMES * SAMPLES_PER_FRAME
        pad_r = (-len(wav) % SAMPLES_PER_FRAME) + (delay + 1 + RIGHT_PAD_BUFFER_FRAMES) * SAMPLES_PER_FRAME
        x = np.concatenate([np.zeros(pad_l, np.float32), wav, np.zeros(pad_r, np.float32)])
        fe, step, outs = Frontend(self), int(chunk_s * SAMPLE_RATE), []
        for s in range(0, len(x), step):
            e = fe.feed(x[s:s + step])
            if e is not None:
                outs.append(e)
        return torch.cat(outs, 0)

    def transcribe_ids(self, audio, delay_ms: float | None = None) -> list[int]:
        self.load()
        wav = load_audio(audio) if isinstance(audio, (str, os.PathLike)) else to_float32(audio)
        d = self.delay_frames(delay_ms)
        return greedy_decode(self, self.encode(wav, d), d)

    def decode_text(self, ids: list[int]) -> str:
        ids = self.guard.strip_incomplete([t for t in ids if t not in NON_TEXT_TOKENS])
        return b"".join(self.token_bytes[t] for t in ids).decode("utf-8", errors="replace").strip()

    def transcribe(self, audio, delay_ms: float | None = None) -> str:
        """Greedy transcription of a file path or a 16 kHz mono float32/int16 array."""
        return self.decode_text(self.transcribe_ids(audio, delay_ms))


class _GraphStep:
    """CUDA graph of one decoder step bound to a cache's static buffers."""

    def __init__(self, model: Voxtral, cache: DecoderCache, ada: torch.Tensor, span: int):
        dev = model.device
        self.audio = torch.zeros(cache.batch, 1, model.cfg.dec_dim, device=dev, dtype=model.dtype)
        self.tokens = torch.zeros(cache.batch, 1, dtype=torch.long, device=dev)
        self.ada, self.ada_src = ada.clone(), ada
        fn = model._step if model._step is not None else model.net.decoder
        args = (self.audio, self.tokens, cache.pos, self.ada, cache.k, cache.v, cache.kpos, span)
        # the warmup writes junk K/V at the current position; the real step at that position
        # rewrites it before attending, so capturing mid-sequence is safe
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(*args)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(*args)

    def run(self, audio, tokens, ada):
        if ada is not self.ada_src:
            self.ada.copy_(ada)
            self.ada_src = ada
        self.audio.copy_(audio)
        self.tokens.copy_(tokens)
        self.graph.replay()
        return self.out


def load_model(path: str, device: str = "auto", dtype: str = "auto", **kw) -> Voxtral:
    return Voxtral(path, device=device, dtype=dtype, **kw)

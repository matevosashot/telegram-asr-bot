"""Audio loading and the causal, chunkable front end (mel -> conv stem).

Chunked and whole-signal processing give identical frames: the mel keeps a short
audio tail and the convolutions are causal with zero left padding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (GLOBAL_LOG_MEL_MAX, HOP_LENGTH, NUM_MEL_BINS,
                        SAMPLE_RATE, WINDOW_SIZE)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any soundfile-readable file as mono float32 at ``sr``."""
    import soundfile as sf
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        import soxr
        wav = soxr.resample(wav, file_sr, sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def to_float32(pcm) -> np.ndarray:
    """int16 / float PCM (numpy array, list or bytes-like) -> float32 in [-1, 1]."""
    a = np.asarray(pcm)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    return np.ascontiguousarray(a, dtype=np.float32)


def _hertz_to_mel(freq):
    min_log_hertz, min_log_mel = 1000.0, 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels


def _mel_to_hertz(mels):
    min_log_hertz, min_log_mel = 1000.0, 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def mel_filters() -> torch.Tensor:
    """Slaney mel filter bank, [201, 128] float32."""
    n_bins = 1 + WINDOW_SIZE // 2
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, n_bins)
    mel_freqs = np.linspace(_hertz_to_mel(0.0), _hertz_to_mel(8000.0), NUM_MEL_BINS + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)
    diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down, up))
    fb *= np.expand_dims(2.0 / (filter_freqs[2:NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS]), 0)
    return torch.from_numpy(fb).float()


class MelStream:
    """Incremental log-mel; state is a short audio tail (starts with 200 zeros)."""

    _MAX_FRAMES = 4096  # bound the [frames, 400] window tensor

    def __init__(self, filters: torch.Tensor, device):
        self.device = device
        self.filters = filters.to(device)
        self.window = torch.hann_window(WINDOW_SIZE, device=device)
        self.floor = GLOBAL_LOG_MEL_MAX - 8.0
        self.reset()

    def reset(self):
        self.buf = torch.zeros(WINDOW_SIZE // 2, dtype=torch.float32, device=self.device)

    def feed(self, audio: torch.Tensor):
        """audio: float32 [N] on ``device``. Returns [128, M] or None."""
        self.buf = torch.cat([self.buf, audio])
        n_new = (self.buf.shape[0] - WINDOW_SIZE) // HOP_LENGTH
        if n_new <= 0:
            return None
        outs = []
        for s in range(0, n_new, self._MAX_FRAMES):
            e = min(n_new, s + self._MAX_FRAMES)
            seg = self.buf[s * HOP_LENGTH:(e - 1) * HOP_LENGTH + WINDOW_SIZE]
            frames = seg.unfold(0, WINDOW_SIZE, HOP_LENGTH) * self.window
            mag = torch.fft.rfft(frames, n=WINDOW_SIZE).abs() ** 2
            spec = (mag @ self.filters).clamp(min=1e-10).log10()
            spec = torch.clamp(spec, min=self.floor)
            outs.append(((spec + 4.0) / 4.0).T)
        self.buf = self.buf[n_new * HOP_LENGTH:]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


class CausalConv1dStream:
    """Causal conv1d with an inter-chunk input buffer (zero left padding)."""

    def __init__(self, weight, bias, stride):
        self.weight, self.bias, self.stride = weight, bias, stride
        self.kernel = weight.shape[2]
        self.pad = self.kernel - stride
        self.reset()

    def reset(self):
        w = self.weight
        self.buf = torch.zeros(1, w.shape[1], self.pad, device=w.device, dtype=w.dtype)

    def feed(self, x):
        self.buf = torch.cat([self.buf, x], dim=-1)
        L = self.buf.shape[-1]
        if L < self.kernel:
            return None
        n_out = (L - self.kernel) // self.stride + 1
        usable = (n_out - 1) * self.stride + self.kernel
        out = F.conv1d(self.buf[:, :, :usable], self.weight, self.bias, stride=self.stride)
        self.buf = self.buf[:, :, n_out * self.stride:]
        return out


class ConvStem:
    """conv1(k3,s1) -> GELU -> conv2(k3,s2) -> GELU; mel [128, N] -> tokens [M, 1280]."""

    def __init__(self, w1, b1, w2, b2):
        self.c1 = CausalConv1dStream(w1, b1, 1)
        self.c2 = CausalConv1dStream(w2, b2, 2)

    def reset(self):
        self.c1.reset()
        self.c2.reset()

    def feed(self, mel):
        h = self.c1.feed(mel.unsqueeze(0))
        if h is None:
            return None
        h = self.c2.feed(F.gelu(h))
        if h is None:
            return None
        return F.gelu(h).squeeze(0).transpose(0, 1)

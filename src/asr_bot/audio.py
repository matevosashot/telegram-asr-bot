"""Audio helpers: probing and converting any input to 16 kHz mono PCM via ffmpeg."""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    pass


def probe_duration(path: Path | str) -> float | None:
    """Duration in seconds according to ffprobe, or None if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=60,
        ).stdout
        value = json.loads(out).get("format", {}).get("duration")
        return float(value) if value not in (None, "N/A") else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def convert_to_wav(src: Path | str, dst: Path | str, max_seconds: float | None = None) -> Path:
    """Decode any audio/video file to 16 kHz mono 16-bit WAV (first audio stream only)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-map", "0:a:0",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise AudioError("Could not decode audio: " + (msg[-1] if msg else "ffmpeg failed"))
    return Path(dst)


def read_wav(path: Path | str) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise AudioError("expected 16 kHz mono 16-bit WAV")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path | str, samples: np.ndarray) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return Path(path)


@dataclass
class Audio:
    """16 kHz mono float32 audio, with a WAV file on disk for backends that want a path."""

    samples: np.ndarray
    path: Path | None = None
    _tmp: list[Path] = field(default_factory=list, repr=False)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @classmethod
    def from_file(cls, path: Path | str, workdir: Path | str | None = None,
                  max_seconds: float | None = None) -> "Audio":
        """Decode any media file (at most ``max_seconds`` of it). The 16 kHz WAV is written
        next to it (or into ``workdir``)."""
        path = Path(path)
        out_dir = Path(workdir) if workdir else path.parent
        wav = convert_to_wav(path, out_dir / (path.stem + ".16k.wav"), max_seconds)
        return cls(read_wav(wav), wav)

    def ensure_path(self) -> Path:
        if self.path is None:
            fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            fd.close()
            self.path = write_wav(fd.name, self.samples)
            self._tmp.append(self.path)
        return self.path

    def head(self, seconds: float) -> "Audio":
        n = int(seconds * SAMPLE_RATE)
        if n >= len(self.samples):
            return self
        return Audio(self.samples[:n])

    def is_silent(self, peak_threshold: float = 3e-4) -> bool:
        """True for empty or digitally silent audio (peak below about -70 dBFS). Quiet but real
        recordings (peak around -45 dBFS are common) must still pass."""
        return len(self.samples) == 0 or float(np.max(np.abs(self.samples))) < peak_threshold

    def cleanup(self) -> None:
        for p in self._tmp:
            p.unlink(missing_ok=True)
        self._tmp.clear()


def chunk(samples: np.ndarray, max_seconds: float, overlap_seconds: float = 0.0) -> list[np.ndarray]:
    """Split audio into chunks of at most ``max_seconds`` (cutting at the quietest point
    within the last 20% of each window, to avoid splitting words)."""
    size = int(max_seconds * SAMPLE_RATE)
    if len(samples) <= size:
        return [samples]
    chunks = []
    start = 0
    frame = int(0.02 * SAMPLE_RATE)
    while start < len(samples):
        end = min(start + size, len(samples))
        if end < len(samples):
            search_from = start + int(size * 0.8)
            region = samples[search_from:end]
            n_frames = len(region) // frame
            if n_frames > 1:
                energy = (region[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
                end = search_from + int(np.argmin(energy)) * frame + frame // 2
        chunks.append(samples[start:end])
        if end >= len(samples):
            break
        start = max(end - int(overlap_seconds * SAMPLE_RATE), start + 1)
    return chunks

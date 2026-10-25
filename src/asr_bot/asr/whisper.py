"""OpenAI Whisper via faster-whisper (CTranslate2)."""

from __future__ import annotations

import threading
from typing import Any

from ..audio import SAMPLE_RATE, Audio, chunk
from .. import languages
from .base import ASRBackend, BackendUnavailable, Segment, Transcript, register_backend

# Models are shared between engines/detectors that use the same weights (e.g. the
# "whisper" engine and the whisper language detector), with reference counting.
_models: dict[tuple, list[Any]] = {}  # key -> [model, refcount]
_lock = threading.Lock()


def acquire_model(model: str, device: str, compute_type: str, **kwargs: Any):
    from faster_whisper import WhisperModel

    key = (model, device, compute_type)
    with _lock:
        entry = _models.get(key)
        if entry is None:
            entry = [WhisperModel(model, device=device, compute_type=compute_type, **kwargs), 0]
            _models[key] = entry
        entry[1] += 1
        return entry[0]


def release_model(model: str, device: str, compute_type: str) -> None:
    from .base import free_torch_memory

    key = (model, device, compute_type)
    with _lock:
        entry = _models.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            del _models[key]
    free_torch_memory()


def whisper_languages() -> set[str | None]:
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return {None, *_LANGUAGE_CODES}
    except ImportError:  # pragma: no cover
        return {None}


def check_faster_whisper() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as e:
        raise BackendUnavailable("faster-whisper is not installed (pip install 'asr-bot[whisper]')") from e


@register_backend("faster-whisper")
class FasterWhisperBackend(ASRBackend):
    display_name = "Whisper"
    supported_languages = None
    detects_language = True
    can_score = True

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        self.compute_type = options.get("compute_type", "float16" if device.startswith("cuda") else "int8")
        self._model = None

    @classmethod
    def check_available(cls) -> None:
        check_faster_whisper()

    def load(self) -> None:
        if self._model is None:
            self._model = acquire_model(self.model_name or "large-v3", self.device, self.compute_type)

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            release_model(self.model_name or "large-v3", self.device, self.compute_type)

    def is_loaded(self) -> bool:
        return self._model is not None

    def supports(self, language: str | None) -> bool:
        if language is not None and language not in whisper_languages():
            return False
        return super().supports(language)

    def _single_language(self) -> str | None:
        langs = self.supported_languages
        return next(iter(langs)) if langs and len(langs) == 1 else None

    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        self.load()
        language = language or self._single_language()
        if language not in whisper_languages():
            language = None
        # Whisper's decoder emits a limited number of tokens per window, and scripts like
        # Armenian need several byte-level tokens per character, so long windows get cut
        # off. ``max_chunk_seconds`` splits the audio at quiet points first.
        max_chunk = float(self.options.get("max_chunk_seconds", 0) or 0)
        pieces = chunk(audio.samples, max_chunk) if max_chunk else [audio.samples]
        segs: list[Segment] = []
        info = None
        offset = 0.0
        for piece in pieces:
            segments, piece_info = self._model.transcribe(
                piece,
                language=language,
                beam_size=int(self.options.get("beam_size", 5)),
                vad_filter=bool(self.options.get("vad_filter", not max_chunk)),
                condition_on_previous_text=bool(self.options.get("condition_on_previous_text", False)),
            )
            segs += [Segment(offset + s.start, offset + s.end, s.text.strip()) for s in segments]
            info = info or piece_info
            language = language or piece_info.language  # keep the first piece's language
            offset += len(piece) / SAMPLE_RATE
        return Transcript(
            text=" ".join(s.text for s in segs if s.text).strip(),
            language=languages.normalize(info.language) if info else language,
            language_prob=info.language_probability if info else None,
            segments=segs,
        )

    def score(self, audio: Audio, language: str) -> float:
        self.load()
        segments, _ = self._model.transcribe(
            audio.samples, language=language, beam_size=1, without_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, temperature=0.0,
        )
        total, weight = 0.0, 0.0
        for s in segments:
            w = max(s.end - s.start, 0.1)
            total += s.avg_logprob * w
            weight += w
        return total / weight if weight else float("-inf")

    def detect_language(self, audio: Audio) -> dict[str, float]:
        """Full language probability distribution (used by the whisper language detector)."""
        self.load()
        _, _, all_probs = self._model.detect_language(audio.samples)
        out: dict[str, float] = {}
        for code, p in all_probs:
            c = languages.normalize(code)
            if c:
                out[c] = out.get(c, 0.0) + float(p)
        return out

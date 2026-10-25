"""Language identification with Whisper's language token probabilities (faster-whisper)."""

from __future__ import annotations

from typing import Any

from ..asr.whisper import FasterWhisperBackend, check_faster_whisper
from ..audio import Audio
from .base import LanguageDetector, register_detector


@register_detector("whisper-lid")
class WhisperLID(LanguageDetector):
    display_name = "Whisper LID"
    window_seconds = 30.0

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        opts = {"model": options.get("model", "large-v3-turbo"), **options}
        self._whisper = FasterWhisperBackend(f"lid:{name}", opts, device)

    @classmethod
    def check_available(cls) -> None:
        check_faster_whisper()

    def load(self) -> None:
        self._whisper.load()

    def unload(self) -> None:
        self._whisper.unload()

    def is_loaded(self) -> bool:
        return self._whisper.is_loaded()

    def probabilities(self, audio: Audio) -> dict[str, float]:
        return self._whisper.detect_language(audio)

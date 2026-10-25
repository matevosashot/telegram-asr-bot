"""Spoken language identification, independent of the ASR engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from .. import languages
from ..audio import Audio


@dataclass
class LanguageGuess:
    language: str | None
    probability: float
    # Top alternatives (after restriction to candidate languages), best first.
    top: list[tuple[str, float]] = field(default_factory=list)
    method: str = ""

    @classmethod
    def from_probs(cls, probs: dict[str, float], candidates: list[str], method: str) -> "LanguageGuess":
        restricted = languages.restrict(probs, candidates)
        ranked = sorted(restricted.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return cls(None, 0.0, [], method)
        return cls(ranked[0][0], float(ranked[0][1]), [(k, float(v)) for k, v in ranked[:5]], method)


class LanguageDetector(ABC):
    display_name: ClassVar[str] = "LID"
    #: Seconds of audio from the beginning used for detection.
    window_seconds: ClassVar[float] = 30.0

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        self.name = name
        self.options = options
        self.device = device

    @classmethod
    def check_available(cls) -> None:
        """Raise asr_bot.asr.BackendUnavailable if dependencies are missing."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def probabilities(self, audio: Audio) -> dict[str, float]:
        """Probability per ISO-639-1 language code."""

    def detect(self, audio: Audio, candidates: list[str]) -> LanguageGuess:
        window = float(self.options.get("window_seconds", self.window_seconds))
        probs = self.probabilities(audio.head(window))
        return LanguageGuess.from_probs(probs, candidates, self.name)


_REGISTRY: dict[str, type[LanguageDetector]] = {}


def register_detector(type_name: str) -> Callable[[type[LanguageDetector]], type[LanguageDetector]]:
    def deco(cls: type[LanguageDetector]) -> type[LanguageDetector]:
        _REGISTRY[type_name] = cls
        return cls
    return deco


_loaded = False


def detector_types() -> dict[str, type[LanguageDetector]]:
    global _loaded
    if not _loaded:
        _loaded = True
        from . import voxlingua, whisper_lid  # noqa: F401

        from importlib.metadata import entry_points

        for ep in entry_points(group="asr_bot.detectors"):
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, LanguageDetector):
                _REGISTRY.setdefault(ep.name, obj)
    return dict(_REGISTRY)

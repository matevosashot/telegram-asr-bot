"""The ASR engine interface. Implement ``ASRBackend`` to add a new engine.

Minimal example (see README, "Adding your own engine")::

    from asr_bot.asr import ASRBackend, Transcript, register_backend

    @register_backend("my-asr")
    class MyASR(ASRBackend):
        display_name = "My ASR"
        supported_languages = {"hy"}

        def load(self):
            self.model = ...            # self.options holds the [engines.<name>] table

        def unload(self):
            self.model = None

        def is_loaded(self):
            return getattr(self, "model", None) is not None

        def transcribe(self, audio, language=None):
            text = self.model.run(audio.samples)   # 16 kHz mono float32 numpy array
            return Transcript(text=text, language=language or "hy")
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from ..audio import Audio

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    # Language the text is in (ISO-639-1) as reported by the engine, or the hint it was given.
    language: str | None = None
    language_prob: float | None = None
    segments: list[Segment] = field(default_factory=list)
    # Filled in by the engine manager:
    engine: str = ""
    duration: float = 0.0
    elapsed: float = 0.0


class BackendUnavailable(Exception):
    """The backend's dependencies are not installed or it can't run on this machine."""


class ASRBackend(ABC):
    """An ASR engine instance, configured by an ``[engines.<name>]`` config table.

    Methods are synchronous; the engine manager calls them from a single worker thread,
    so implementations don't need to be thread-safe.
    """

    #: Human-readable engine family name.
    display_name: ClassVar[str] = "ASR"
    #: ISO-639-1 codes the engine supports; ``None`` means "practically any language".
    supported_languages: ClassVar[set[str] | None] = None
    #: True if the engine detects the language itself when no hint is given.
    detects_language: ClassVar[bool] = False
    #: True if ``score()`` is implemented (used by language_detector = "engine-scoring").
    can_score: ClassVar[bool] = False

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        self.name = name
        self.options = options
        self.device = device
        langs = options.get("languages")
        if langs:
            # Config may narrow (or declare) the languages, e.g. for a fine-tuned model.
            self.supported_languages = {str(x).lower() for x in langs}

    @classmethod
    def check_available(cls) -> None:
        """Raise BackendUnavailable if dependencies are missing. Must be cheap (no model load)."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        """Transcribe 16 kHz mono audio. ``language`` is an ISO-639-1 hint or None."""

    def score(self, audio: Audio, language: str) -> float:
        """How well ``audio`` matches ``language`` (higher is better, e.g. mean token log-prob)."""
        raise NotImplementedError

    def supports(self, language: str | None) -> bool:
        return language is None or self.supported_languages is None or language in self.supported_languages

    @property
    def model_name(self) -> str:
        return str(self.options.get("model", ""))


_REGISTRY: dict[str, type[ASRBackend]] = {}


def register_backend(type_name: str) -> Callable[[type[ASRBackend]], type[ASRBackend]]:
    def deco(cls: type[ASRBackend]) -> type[ASRBackend]:
        _REGISTRY[type_name] = cls
        return cls
    return deco


def backend_types() -> dict[str, type[ASRBackend]]:
    _load_builtin_and_plugins()
    return dict(_REGISTRY)


_loaded = False


def _load_builtin_and_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import omnilingual, qwen, voxtral, whisper  # noqa: F401  (register built-ins)

    from importlib.metadata import entry_points

    for ep in entry_points(group="asr_bot.backends"):
        try:
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, ASRBackend) and ep.name not in _REGISTRY:
                _REGISTRY[ep.name] = obj
        except Exception:  # pragma: no cover - plugin problems must not kill the bot
            log.exception("failed to load ASR backend plugin %s", ep.name)


def free_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

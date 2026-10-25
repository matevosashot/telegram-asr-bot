"""Voxtral Mini 4B Realtime (https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

A pure-PyTorch runtime: no ``transformers`` at run time, a one-time conversion of the checkpoint into
a fused flat file for fast loading, CUDA-graph decoding and optional ``torch.compile``. Adapted from the
``voxtral-offline`` package (Apache-2.0).

Use the original release (downloaded from Hugging Face on first use) or any checkpoint directory with the
same architecture (``checkpoint = "/path"``): ``config.json``, ``model.safetensors`` and ``tekken.json``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...audio import Audio
from ..base import ASRBackend, BackendUnavailable, Transcript, free_torch_memory, register_backend

log = logging.getLogger(__name__)

ORIGINAL_REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CHECKPOINT_FILES = ("config.json", "model.safetensors", "tekken.json")
# Languages listed on the model card (ISO-639-1). The model takes no language hint.
VOXTRAL_LANGUAGES = {"en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko"}


@register_backend("voxtral")
class VoxtralBackend(ASRBackend):
    display_name = "Voxtral Realtime"
    supported_languages = VOXTRAL_LANGUAGES
    detects_language = False

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        self._model = None

    @classmethod
    def check_available(cls) -> None:
        try:
            import mistral_common  # noqa: F401  (tokenizer table, needed once per checkpoint)
            import safetensors  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"{e.name} is not installed (pip install 'asr-bot[voxtral]')") from e

    @property
    def model_name(self) -> str:
        # Shown in menus; don't expose the local path of a custom checkpoint.
        return "custom checkpoint" if self.options.get("checkpoint") else ORIGINAL_REPO

    # ---- checkpoint --------------------------------------------------------------------

    @property
    def delay_ms(self) -> float:
        return float(self.options.get("delay_ms", 2400))

    def _cache_dir(self) -> str:
        from . import cache

        return cache.cache_dir(self.options.get("cache_dir"))

    def _original(self, files: tuple[str, ...] = CHECKPOINT_FILES) -> str:
        """Directory holding ``files`` of the original release; downloads what is missing (once)."""
        d = os.path.join(self._cache_dir(), "original")
        if not all(os.path.exists(os.path.join(d, f)) for f in files):
            from huggingface_hub import snapshot_download

            log.info("downloading %s (%s) to %s", ORIGINAL_REPO, ", ".join(files), d)
            snapshot_download(ORIGINAL_REPO, local_dir=d, allow_patterns=list(files))
        return d

    def resolve_checkpoint(self) -> tuple[str, str | None]:
        """(checkpoint directory, tokenizer file to use instead of the directory's own, or None)."""
        custom = self.options.get("checkpoint")
        if not custom:
            return self._original(), None
        path = os.path.abspath(os.path.expanduser(str(custom)))
        for f in ("config.json", "model.safetensors"):
            if not os.path.exists(os.path.join(path, f)):
                raise FileNotFoundError(f"voxtral checkpoint {path!r} has no {f}")
        if os.path.exists(os.path.join(path, "tekken.json")):
            return path, None
        return path, os.path.join(self._original(("tekken.json",)), "tekken.json")

    # ---- lifecycle ---------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            self._model.load()  # re-materialise after unload(); a no-op when loaded
            return
        from .model import Voxtral

        path, tekken = self.resolve_checkpoint()
        model = Voxtral(path, device=self.device or "auto", dtype=str(self.options.get("dtype", "auto")),
                        cache_dir=self.options.get("cache_dir"), tekken=tekken)
        model.delay_frames(self.delay_ms)  # reject a bad delay_ms before the first request
        if self.options.get("compile") and model.device.type == "cuda":
            model.compile()
        self._model = model

    def unload(self) -> None:
        if self._model is not None:
            self._model.offload()
        free_torch_memory()

    def is_loaded(self) -> bool:
        return self._model is not None and self._model.loaded

    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        self.load()
        text = self._model.transcribe(audio.samples, delay_ms=self.delay_ms)
        return Transcript(text=text, language=language if language and self.supports(language) else None)

"""Qwen3-ASR (https://github.com/QwenLM/Qwen3-ASR) via the ``qwen-asr`` package."""

from __future__ import annotations

from typing import Any

from .. import languages
from ..audio import SAMPLE_RATE, Audio
from .base import ASRBackend, BackendUnavailable, Transcript, free_torch_memory, register_backend

# Languages listed in the Qwen3-ASR model card (ISO-639-1). No Armenian.
QWEN3_LANGUAGES = {
    "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th", "vi", "ja",
    "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "hu", "mk", "ro",
}


@register_backend("qwen3-asr")
class Qwen3ASRBackend(ASRBackend):
    display_name = "Qwen3-ASR"
    supported_languages = QWEN3_LANGUAGES
    detects_language = True

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        self._model = None

    @classmethod
    def check_available(cls) -> None:
        try:
            import qwen_asr  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable("qwen-asr is not installed (pip install 'asr-bot[qwen]')") from e

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ASRModel

        device_map = "cuda:0" if self.device == "cuda" else self.device
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name or "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16 if device_map.startswith("cuda") else torch.float32,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=int(self.options.get("max_new_tokens", 4096)),
        )

    def unload(self) -> None:
        self._model = None
        free_torch_memory()

    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: Audio, language: str | None = None) -> Transcript:
        self.load()
        hint = languages.name(language) if language and language in QWEN3_LANGUAGES else None
        result = self._model.transcribe(audio=(audio.samples, SAMPLE_RATE), language=hint)[0]
        reported = (getattr(result, "language", "") or "").split(",")[0].strip()  # e.g. "Chinese,English"
        detected = languages.normalize(reported or hint)
        return Transcript(text=(result.text or "").strip(), language=detected)

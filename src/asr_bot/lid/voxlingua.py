"""SpeechBrain VoxLingua107 ECAPA language identification (107 languages, incl. Armenian)."""

from __future__ import annotations

from typing import Any

from .. import languages
from ..asr.base import BackendUnavailable, free_torch_memory
from ..audio import Audio
from .base import LanguageDetector, register_detector


@register_detector("voxlingua107")
class VoxLinguaLID(LanguageDetector):
    display_name = "VoxLingua107"
    window_seconds = 20.0

    def __init__(self, name: str, options: dict[str, Any], device: str = "cuda"):
        super().__init__(name, options, device)
        self._model = None
        self._labels: list[str | None] | None = None

    @classmethod
    def check_available(cls) -> None:
        try:
            import speechbrain  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable("speechbrain is not installed (pip install 'asr-bot[voxlingua]')") from e

    def load(self) -> None:
        if self._model is not None:
            return
        from speechbrain.inference.classifiers import EncoderClassifier

        source = self.options.get("model", "speechbrain/lang-id-voxlingua107-ecapa")
        self._model = EncoderClassifier.from_hparams(
            source=source,
            savedir=self.options.get("savedir"),
            run_opts={"device": "cuda:0" if self.device == "cuda" else self.device},
        )
        encoder = self._model.hparams.label_encoder
        n = len(encoder.ind2lab)
        self._labels = [languages.normalize(encoder.ind2lab[i]) for i in range(n)]

    def unload(self) -> None:
        self._model = None
        free_torch_memory()

    def is_loaded(self) -> bool:
        return self._model is not None

    def probabilities(self, audio: Audio) -> dict[str, float]:
        import torch

        self.load()
        wav = torch.from_numpy(audio.samples).unsqueeze(0)
        with torch.no_grad():
            out_prob, _, _, _ = self._model.classify_batch(wav)
        probs = torch.softmax(out_prob[0].float(), dim=-1).cpu().numpy()
        result: dict[str, float] = {}
        for label, p in zip(self._labels or [], probs):
            if label:
                result[label] = result.get(label, 0.0) + float(p)
        return result

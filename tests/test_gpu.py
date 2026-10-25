"""Real engines on a few FLEURS test clips. Run with:  pytest -m gpu -s

Clips are fetched once into tests/data/cache/fleurs by scripts/fetch_fleurs.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from asr_bot.audio import Audio
from asr_bot.config import config_from_dict
from asr_bot.engine import EngineManager, Plan

pytestmark = pytest.mark.gpu

DATA = Path(__file__).parent / "data" / "cache" / "fleurs"
LANGS = {"hy_am": "hy", "ru_ru": "ru", "en_us": "en"}
N_CLIPS = int(os.environ.get("ASR_BOT_TEST_CLIPS", 3))


def _clips():
    if not DATA.exists():
        pytest.skip("no FLEURS clips; run scripts/fetch_fleurs.py first")
    out = []
    for fleurs, code in LANGS.items():
        refs = json.loads((DATA / f"{fleurs}.json").read_text())
        out += [(DATA / name, code, ref) for name, ref in list(refs.items())[:N_CLIPS]]
    return out


def _work() -> Path:
    d = DATA / "work"
    d.mkdir(exist_ok=True)
    return d


def cer(ref: str, hyp: str) -> float:
    # Armenian "և" is the ligature of "եւ"; models differ in which they write.
    norm = lambda s: re.sub(r"[^\w]+", "", s.lower().replace("և", "եւ"))  # noqa: E731
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


@pytest.fixture(scope="module")
def manager():
    cfg = config_from_dict({
        "telegram": {"owner_id": 1},
        "engines": {
            "whisper": {"type": "faster-whisper", "model": "large-v3"},
            "whisper-hy": {"type": "faster-whisper", "model": "Chillarmo/whisper-large-v3-turbo-hy-ct2",
                           "languages": ["hy"], "max_chunk_seconds": 12},
            "qwen3": {"type": "qwen3-asr", "model": "Qwen/Qwen3-ASR-1.7B"},
            "omnilingual": {"type": "omnilingual", "model": "omniASR_LLM_Unlimited_3B_v2"},
        },
    })
    em = EngineManager(cfg)
    yield em
    em.unload_all()


@pytest.mark.parametrize("detector", ["whisper", "voxlingua", "engine-scoring"])
async def test_language_detection(manager, detector):
    for path, code, _ in _clips():
        audio = Audio.from_file(path, _work())
        guess = manager._detect(audio, Plan("whisper", detector, ["hy", "ru", "en"], 0.5))
        print(f"{detector:15s} {path.name}: {guess.language} {guess.probability:.2f}")
        assert guess.language == code


@pytest.mark.parametrize("engine", ["whisper", "whisper-hy", "qwen3", "omnilingual"])
async def test_engine_quality(manager, engine):
    eng = manager.engines.get(engine)
    if eng is None:
        pytest.skip(f"{engine} unavailable: {manager.unavailable.get(engine)}")
    results = {}
    clips = [c for c in _clips() if eng.supports(c[1])]
    warm = Audio.from_file(clips[0][0], _work())
    await manager.transcribe(warm, Plan(engine, "whisper", ["hy", "ru", "en"], 0.5))  # load the model
    for path, code, ref in clips:
        audio = Audio.from_file(path, _work())
        t0 = time.time()
        r = await manager.transcribe(audio, Plan(engine, "whisper", ["hy", "ru", "en"], 0.5))
        results.setdefault(code, []).append((cer(ref, r.transcript.text), time.time() - t0, audio.duration))
    for code, rows in results.items():
        avg_cer = sum(c for c, _, _ in rows) / len(rows)
        rtf = sum(t for _, t, _ in rows) / sum(d for _, _, d in rows)
        print(f"{engine:12s} {code}: CER {avg_cer:.3f}  RTF {rtf:.3f}")
        assert avg_cer < 0.35, f"{engine} {code} CER too high: {avg_cer:.3f}"

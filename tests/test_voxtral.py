"""Voxtral engine: checkpoint resolution, conversion cache, and the exactness of the chunked front end."""

from __future__ import annotations

import os
import time

import pytest
import torch
from safetensors.torch import save_file

from asr_bot.asr import backend_types
from asr_bot.asr.voxtral import ORIGINAL_REPO, VOXTRAL_LANGUAGES, VoxtralBackend
from asr_bot.asr.voxtral import cache as C
from asr_bot.asr.voxtral.frontend import ConvStem, MelStream, mel_filters
from asr_bot.asr.voxtral.guard import Utf8Guard, incomplete_tail, valid_prefix


def make_checkpoint(path, tekken=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    save_file({"x": torch.zeros(2)}, str(path / "model.safetensors"))
    if tekken:
        (path / "tekken.json").write_text("{}")
    return path


def backend(tmp_path, **options) -> VoxtralBackend:
    return VoxtralBackend("voxtral", {"cache_dir": str(tmp_path / "cache"), **options}, "cpu")


def test_registered_and_languages(tmp_path):
    assert backend_types()["voxtral"] is VoxtralBackend
    b = backend(tmp_path)
    assert b.supported_languages == VOXTRAL_LANGUAGES and b.supports("ru") and not b.supports("hy")
    assert backend(tmp_path, languages=["hy", "en"]).supports("hy")  # a fine-tune can declare its own


def test_custom_checkpoint(tmp_path):
    ckpt = make_checkpoint(tmp_path / "my-model")
    assert backend(tmp_path, checkpoint=str(ckpt)).resolve_checkpoint() == (str(ckpt), None)


def test_custom_checkpoint_must_be_complete(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors|config.json"):
        backend(tmp_path, checkpoint=str(tmp_path / "empty")).resolve_checkpoint()


def test_missing_tokenizer_comes_from_the_original(tmp_path, monkeypatch):
    ckpt = make_checkpoint(tmp_path / "no-tok", tekken=False)
    b = backend(tmp_path, checkpoint=str(ckpt))
    asked = []
    monkeypatch.setattr(b, "_original", lambda files=(): asked.append(files) or str(tmp_path / "orig"))
    path, tekken = b.resolve_checkpoint()
    assert path == str(ckpt) and tekken == str(tmp_path / "orig" / "tekken.json")
    assert asked == [("tekken.json",)]  # only the 15 MB tokenizer, not the 9 GB weights


def test_default_is_the_original_release_and_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, local_dir=None, allow_patterns=None, **kw):
        calls.append((repo, sorted(allow_patterns)))
        make_checkpoint(type(tmp_path)(local_dir))
        return local_dir

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    b = backend(tmp_path)
    path, tekken = b.resolve_checkpoint()
    assert path == str(tmp_path / "cache" / "original") and tekken is None
    assert calls == [(ORIGINAL_REPO, ["config.json", "model.safetensors", "tekken.json"])]
    b.resolve_checkpoint()
    assert len(calls) == 1  # files are there now: no network


def test_guard_utf8():
    assert incomplete_tail("é".encode()[:1]) == b"\xc3"
    assert incomplete_tail("abc".encode()) == b""
    assert valid_prefix("ab".encode() + "բ".encode()[:1]) and not valid_prefix(b"\x80")
    assert not valid_prefix(b"\xc0\x80") and not valid_prefix(b"\xff")
    tb = [b"", b"a", "բ".encode()[:1], "բ".encode()[1:], b"\x80", b"\xc3"]
    g = Utf8Guard(tb, "cpu")
    b0 = g.bias_for(b"")
    assert b0[1] == 0 and b0[0] == 0 and b0[2] == 0 and b0[3] == float("-inf") and b0[4] == float("-inf")
    pend = g.advance(b"", 2)
    b1 = g.bias_for(pend)
    assert b1[3] == 0 and b1[1] == float("-inf") and b1[0] == float("-inf")
    assert g.strip_incomplete([1, 2]) == [1] and g.strip_incomplete([1, 2, 3]) == [1, 2, 3]


def test_cache_freshness(tmp_path):
    root = str(tmp_path / "cache")
    src = make_checkpoint(tmp_path / "model-dir")
    sf = str(src / "model.safetensors")
    out = C.cached_path(str(src), "bf16", root)
    assert out != C.cached_path(str(src), "fp32", root)
    assert out != C.cached_path(str(src), "bf16", root, tekken="/other/tekken.json")
    assert not C.is_fresh(out, sf, "bf16")
    st = os.stat(sf)
    save_file({"y": torch.zeros(1)}, out, metadata={"converter_version": C.CONVERTER_VERSION, "dtype": "bf16",
                                                   "source_size": str(st.st_size), "source_mtime_ns": str(st.st_mtime_ns)})
    assert C.is_fresh(out, sf, "bf16")
    assert not C.is_fresh(out, sf, "fp32")  # dtype mismatch in the metadata
    time.sleep(0.05)
    os.utime(sf)  # source touched: the cached file is stale
    assert not C.is_fresh(out, sf, "bf16")


def test_mel_chunking_is_exact():
    x = torch.randn(16000 * 3)
    whole = MelStream(mel_filters(), "cpu").feed(x)
    b = MelStream(mel_filters(), "cpu")
    parts = [p for p in (b.feed(x[i:i + 3777]) for i in range(0, len(x), 3777)) if p is not None]
    got = torch.cat(parts, 1)
    assert got.shape == whole.shape and torch.allclose(got, whole, atol=1e-5)


def test_conv_stem_chunking_is_exact():
    g = torch.Generator().manual_seed(0)
    w1, b1 = torch.randn(8, 128, 3, generator=g), torch.randn(8, generator=g)
    w2, b2 = torch.randn(8, 8, 3, generator=g), torch.randn(8, generator=g)
    mel = torch.randn(128, 301, generator=g)
    whole = ConvStem(w1, b1, w2, b2).feed(mel)
    s = ConvStem(w1, b1, w2, b2)
    parts = [p for p in (s.feed(mel[:, i:i + 37]) for i in range(0, 301, 37)) if p is not None]
    got = torch.cat(parts, 0)
    assert got.shape == whole.shape and torch.allclose(got, whole, atol=1e-4)

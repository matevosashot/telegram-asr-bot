"""Conversion cache: ``~/.cache/asr-bot/voxtral`` (the ``cache_dir`` engine option overrides it).

The first load of a source checkpoint in a given dtype converts it once; later loads reuse the file
unless it is stale: the source changed (size or mtime), the converter version changed, or the cached
file is older than the source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct

from .convert import CONVERTER_VERSION, convert

log = logging.getLogger(__name__)


def cache_dir(root: str | None = None) -> str:
    d = os.path.expanduser(root) if root else os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "asr-bot", "voxtral")
    os.makedirs(d, exist_ok=True)
    return d


def _metadata(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)).get("__metadata__", {})


def cached_path(src_dir: str, dtype: str, root: str | None = None, tekken: str | None = None) -> str:
    src = os.path.abspath(src_dir)
    h = hashlib.sha1(f"{src}\0{tekken or ''}".encode()).hexdigest()[:10]
    return os.path.join(cache_dir(root), f"{os.path.basename(src.rstrip('/'))}-{dtype}-{h}.safetensors")


def is_fresh(cache_file: str, src_file: str, dtype: str) -> bool:
    if not os.path.exists(cache_file):
        return False
    st = os.stat(src_file)
    if os.stat(cache_file).st_mtime < st.st_mtime:
        return False
    try:
        md = _metadata(cache_file)
    except Exception:
        return False
    return (md.get("converter_version") == CONVERTER_VERSION and md.get("dtype") == dtype
            and md.get("source_size") == str(st.st_size)
            and md.get("source_mtime_ns") == str(st.st_mtime_ns))


def resolve(src_dir: str, dtype: str, root: str | None = None, tekken: str | None = None) -> str:
    """Path of an up-to-date converted checkpoint for (src_dir, dtype), converting if needed.

    ``tekken``: tokenizer file to use when ``src_dir`` has no ``tekken.json`` of its own."""
    src_file = os.path.join(src_dir, "model.safetensors")
    out = cached_path(src_dir, dtype, root, tekken)
    if is_fresh(out, src_file, dtype):
        return out
    st = os.stat(src_file)
    log.info("converting %s to %s (one time) -> %s", src_file, dtype, out)
    convert(src_dir, out, dtype, meta={"source_size": str(st.st_size),
                                       "source_mtime_ns": str(st.st_mtime_ns)}, tekken=tekken)
    return out

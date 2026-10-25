"""Fast safetensors -> device loader.

``safetensors.load_file(device="cuda")`` copies tensor by tensor from pageable memory
(~1 GB/s). Here the file is streamed through two pinned staging buffers into ONE device
buffer (copy of chunk i overlaps the read of chunk i+1) and tensors are views of it, so
a warm-page-cache load runs at PCIe speed. On CPU the file is simply memory-mapped.
"""
from __future__ import annotations

import json
import struct

import torch

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}
_CHUNK = 256 << 20


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_flat(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    if device.type != "cuda":
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    header, base = read_header(path)
    header.pop("__metadata__", None)
    total = max(v["data_offsets"][1] for v in header.values())
    buf = torch.empty(total, dtype=torch.uint8, device=device)
    pins = [torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    events: list = [None, None]
    with open(path, "rb", buffering=0) as f:
        f.seek(base)
        off, i = 0, 0
        while off < total:
            n = min(_CHUNK, total - off)
            if events[i] is not None:
                events[i].synchronize()          # staging buffer free again
            got = f.readinto(memoryview(pins[i].numpy())[:n])
            if got != n:
                raise IOError("short read")
            buf[off:off + n].copy_(pins[i][:n], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            events[i] = ev
            off += n
            i ^= 1
    torch.cuda.synchronize()
    out = {}
    for name, meta in header.items():
        s, e = meta["data_offsets"]
        out[name] = buf[s:e].view(_DT[meta["dtype"]]).view(meta["shape"])
    return out

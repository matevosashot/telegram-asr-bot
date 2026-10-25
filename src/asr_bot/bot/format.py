"""Rendering transcripts into Telegram messages."""

from __future__ import annotations

import html

from .. import languages
from ..engine import Result
from ..limits import fmt_duration

MAX_MESSAGE = 4096
# Leave room for the header and HTML escaping growth.
CHUNK = 3800
# More chunks than this -> send a .txt file instead.
MAX_CHUNKS = 3


def header(result: Result) -> str:
    lang = result.language
    parts = [f"{languages.flag(lang)} {languages.name(lang)}"]
    g = result.guess
    if g and g.language == lang and g.probability:
        parts[0] += f" {g.probability:.0%}"
    parts.append(result.engine)
    parts.append(fmt_duration(result.transcript.duration))
    return " · ".join(parts)


def split_text(text: str, size: int = CHUNK) -> list[str]:
    """Split on paragraph, then sentence, then word boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > size:
        window = rest[:size]
        cut = -1
        for sep in ("\n\n", "\n", ". ", "! ", "? ", "։ ", " "):
            idx = window.rfind(sep)
            if idx > size // 2:
                cut = idx + len(sep)
                break
        if cut <= 0:
            cut = size
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


def render(result: Result, *, with_header: bool = True) -> list[str]:
    """HTML messages for a result. Empty list means 'too long, send as a file'."""
    text = result.transcript.text.strip()
    head = f"<i>{html.escape(header(result))}</i>" if with_header else ""
    if result.note and with_header:
        head += f"\n<i>⚠️ {html.escape(result.note)}</i>"
    if not text:
        return [(head + "\n\n" if head else "") + "<i>(no speech recognized)</i>"]
    chunks = split_text(text)
    if len(chunks) > MAX_CHUNKS:
        return []
    out = [html.escape(c) for c in chunks]
    if head:
        out[0] = head + "\n\n" + out[0]
    return out


def as_file_text(result: Result) -> str:
    return f"{header(result)}\n\n{result.transcript.text.strip()}\n"

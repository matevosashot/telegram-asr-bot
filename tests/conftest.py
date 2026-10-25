from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from telegram.request import BaseRequest, RequestData

from asr_bot.asr import ASRBackend, Transcript, register_backend
from asr_bot.audio import SAMPLE_RATE, write_wav
from asr_bot.config import config_from_dict
from asr_bot.lid import LanguageDetector, register_detector

OWNER_ID = 1000
BOT_ID = 999
TOKEN = f"{BOT_ID}:TEST"


# ---- fake engine and detector -------------------------------------------------------------


@register_backend("fake")
class FakeBackend(ASRBackend):
    display_name = "Fake"
    detects_language = True
    calls: list[tuple[str, str | None]] = []

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def is_loaded(self):
        return getattr(self, "_loaded", False)

    def transcribe(self, audio, language=None):
        self.load()
        FakeBackend.calls.append((self.name, language))
        text = self.options.get("text", f"hello from {self.name}")
        return Transcript(text=text, language=language or "en")


@register_backend("fake-ru")
class FakeRuBackend(FakeBackend):
    supported_languages = {"ru", "en"}


@register_detector("fake-lid")
class FakeLID(LanguageDetector):
    display_name = "Fake LID"
    probs: dict[str, float] = {"hy": 0.7, "ka": 0.2, "en": 0.1}

    def load(self):
        pass

    def unload(self):
        pass

    def is_loaded(self):
        return False

    def probabilities(self, audio):
        return dict(FakeLID.probs)


def make_config(tmp_path: Path, **overrides: Any):
    data: dict[str, Any] = {
        "telegram": {"owner_id": OWNER_ID},
        "storage": {"data_dir": str(tmp_path / "data")},
        "transcription": {"language_detector": "fake", "candidate_languages": ["hy", "ru", "en"],
                          "idle_unload_minutes": 0, "device": "cpu"},
        "limits": {"per_hour": 2, "per_day": 3, "max_audio_seconds": 60},
        "routing": {"rules": {"hy": "armo"}, "fallback": ["ru", "general"]},
        "engines": {
            "general": {"type": "fake"},
            "armo": {"type": "fake", "text": "բարեւ"},
            "ru": {"type": "fake-ru", "text": "привет"},
        },
        "detectors": {"fake": {"type": "fake-lid"}},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return config_from_dict(data)


@pytest.fixture
def config(tmp_path):
    FakeBackend.calls.clear()
    FakeLID.probs = {"hy": 0.7, "ka": 0.2, "en": 0.1}
    return make_config(tmp_path)


@pytest.fixture
def tone_wav(tmp_path) -> Path:
    t = np.arange(int(SAMPLE_RATE * 2.0)) / SAMPLE_RATE
    return write_wav(tmp_path / "tone.wav", 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32))


@pytest.fixture
def tone_ogg(tmp_path, tone_wav) -> Path:
    out = tmp_path / "tone.ogg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tone_wav), "-c:a", "libopus", str(out)], check=True)
    return out


# ---- fake Telegram transport --------------------------------------------------------------


class FakeTelegram(BaseRequest):
    """Records Bot API calls and returns plausible responses."""

    def __init__(self, file_bytes: bytes = b""):
        self.calls: list[tuple[str, dict]] = []
        self.file_bytes = file_bytes
        self._msg_id = 100
        self.fail_methods: set[str] = set()

    @property
    def read_timeout(self):
        return 5.0

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def methods(self) -> list[str]:
        return [m for m, _ in self.calls]

    def find(self, method: str) -> list[dict]:
        return [p for m, p in self.calls if m == method]

    def texts(self) -> list[str]:
        out = []
        for m, p in self.calls:
            if m in ("sendMessage", "editMessageText"):
                out.append(p.get("text", ""))
        return out

    async def wait_for(self, predicate, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate(self):
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"condition not met; calls: {self.calls}")

    def _message(self, params: dict) -> dict:
        self._msg_id += 1
        chat_id = int(params.get("chat_id", OWNER_ID)) if params.get("chat_id") is not None else OWNER_ID
        return {"message_id": self._msg_id, "date": int(time.time()),
                "chat": {"id": chat_id, "type": "private" if chat_id > 0 else "supergroup"},
                "from": {"id": BOT_ID, "is_bot": True, "first_name": "Bot", "username": "testbot"},
                "text": params.get("text", "")}

    async def do_request(self, url, method, request_data: RequestData | None = None, read_timeout=None,
                         write_timeout=None, connect_timeout=None, pool_timeout=None):
        if "/file/bot" in url:
            return 200, self.file_bytes
        api_method = url.rsplit("/", 1)[-1]
        params = {}
        if request_data is not None:
            params = {k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v)
                      for k, v in (request_data.json_parameters or {}).items()}
        self.calls.append((api_method, params))
        if api_method in self.fail_methods:
            return 400, json.dumps({"ok": False, "error_code": 400,
                                    "description": f"Bad Request: {api_method} failed"}).encode()
        result: Any = True
        if api_method == "getMe":
            result = {"id": BOT_ID, "is_bot": True, "first_name": "Bot", "username": "testbot",
                      "can_join_groups": True, "can_read_all_group_messages": False,
                      "supports_inline_queries": False, "supports_guest_queries": True}
        elif api_method in ("sendMessage", "editMessageText", "sendDocument"):
            result = self._message(params) if "inline_message_id" not in params else True
        elif api_method == "getFile":
            result = {"file_id": params["file_id"], "file_unique_id": "u1", "file_size": len(self.file_bytes),
                      "file_path": "voice/file_1.oga"}
        elif api_method == "answerGuestQuery":
            result = {"inline_message_id": "inline-1"}
        return 200, json.dumps({"ok": True, "result": result}).encode()


@pytest.fixture
def fake_tg(tone_ogg) -> FakeTelegram:
    return FakeTelegram(tone_ogg.read_bytes())


@pytest.fixture
async def app(config, fake_tg):
    from asr_bot.bot.app import _post_init, _post_shutdown, build_application

    application = build_application(config, TOKEN, request=fake_tg)
    await application.initialize()
    await _post_init(application)
    fake_tg.calls.clear()
    yield application
    await _post_shutdown(application)
    await application.shutdown()


# ---- update builders ----------------------------------------------------------------------

_update_id = 0


def _next_id() -> int:
    global _update_id
    _update_id += 1
    return _update_id


def user_dict(user_id: int, username: str | None = None) -> dict:
    d = {"id": user_id, "is_bot": False, "first_name": f"User{user_id}"}
    if username:
        d["username"] = username
    return d


def voice_dict(duration: int = 2, size: int = 5000) -> dict:
    return {"file_id": "voice-file", "file_unique_id": "vf", "duration": duration, "mime_type": "audio/ogg",
            "file_size": size}


def message_dict(user_id: int, *, chat_id: int | None = None, chat_type: str = "private", text: str | None = None,
                 voice: dict | None = None, reply_to: dict | None = None, username: str | None = None,
                 message_id: int = 10, **extra) -> dict:
    chat_id = chat_id if chat_id is not None else user_id
    m: dict[str, Any] = {"message_id": message_id, "date": int(time.time()),
                         "chat": {"id": chat_id, "type": chat_type, **({"title": "G"} if chat_type != "private" else {})},
                         "from": user_dict(user_id, username)}
    if text is not None:
        m["text"] = text
        if text.startswith("/"):
            m["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    if voice is not None:
        m["voice"] = voice
    if reply_to is not None:
        m["reply_to_message"] = reply_to
    m.update(extra)
    return m


async def send_update(app, payload: dict):
    from telegram import Update

    payload = {"update_id": _next_id(), **payload}
    await app.process_update(Update.de_json(payload, app.bot))

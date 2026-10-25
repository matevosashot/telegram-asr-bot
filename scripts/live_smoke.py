"""Live end-to-end check against the real Telegram Bot API.

The bot uploads a voice file to the owner's private chat (so Telegram hosts it and we get a
real file_id), then feeds a synthetic "owner sent this voice" update through the real
Application: download -> ffmpeg -> language ID -> ASR -> reply via the real API.

Usage:
    ASR_BOT_TEST_TOKEN=... python scripts/live_smoke.py -c config.toml AUDIO [AUDIO ...]

It sends a few messages to the owner's chat with the bot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from telegram import Update
from telegram.request import HTTPXRequest

from asr_bot.bot.app import _post_init, _post_shutdown, build_application
from asr_bot.config import load_config


class RecordingRequest(HTTPXRequest):
    def __init__(self) -> None:
        super().__init__(read_timeout=60, write_timeout=120, connect_timeout=20)
        self.calls: list[tuple[str, dict]] = []

    async def do_request(self, url, method, request_data=None, **kw):
        if "/file/bot" not in url:
            params = dict(request_data.json_parameters) if request_data else {}
            self.calls.append((url.rsplit("/", 1)[-1], params))
        return await super().do_request(url, method, request_data, **kw)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("audio", nargs="+", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    token = os.environ["ASR_BOT_TEST_TOKEN"]
    req = RecordingRequest()
    app = build_application(cfg, token, request=req)
    await app.initialize()
    await _post_init(app)
    owner = cfg.telegram.owner_id
    me = await app.bot.get_me()
    ok = True
    try:
        for path in args.audio:
            with path.open("rb") as f:
                sent = await app.bot.send_voice(owner, f, caption=f"🧪 live test: {path.name}")
            req.calls.clear()
            payload = {
                "update_id": int(time.time()),
                "message": {
                    "message_id": sent.message_id, "date": int(time.time()),
                    "chat": {"id": owner, "type": "private"},
                    "from": {"id": owner, "is_bot": False, "first_name": "Owner"},
                    "voice": json.loads(sent.voice.to_json()),
                },
            }
            started = time.time()
            await app.process_update(Update.de_json(payload, app.bot))
            final = None
            while time.time() - started < 900:
                edits = [p for m, p in req.calls if m in ("editMessageText", "sendDocument")]
                if edits and not str(edits[-1].get("text", "")).startswith(("🎧", "⏳ Queued")):
                    final = edits[-1]
                    break
                await asyncio.sleep(0.5)
            print(f"== {path.name} ({time.time() - started:.1f}s) via @{me.username}")
            print(final.get("text") if final else "TIMEOUT")
            ok &= final is not None
    finally:
        await _post_shutdown(app)
        await app.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())

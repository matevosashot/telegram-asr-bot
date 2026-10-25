"""Guest Mode: the bot is summoned by a mention in a chat it is not a member of (e.g. your
1:1 chat with a friend). Reply to a voice message with "@yourbot" and the transcript is
posted into that chat. Enable Guest Mode for the bot in @BotFather first."""

from __future__ import annotations

import html
import logging
import uuid

from telegram import InlineQueryResultArticle, InputTextMessageContent, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..auth import Role
from . import format as fmt
from .services import services
from .transcribe import UserFacingError, check_access, deliver, extract_media, transcribe_media

log = logging.getLogger(__name__)

INLINE_MAX = 4096


def _article(text: str, title: str = "Transcript", parse_mode: str | None = None) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id=uuid.uuid4().hex,
        title=title,
        input_message_content=InputTextMessageContent(text, parse_mode=parse_mode),
    )


async def guest_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    msg = update.guest_message
    if msg is None or not msg.guest_query_id:
        return
    bot = context.bot
    query_id = msg.guest_query_id

    async def answer(text: str, parse_mode: str | None = None) -> str | None:
        try:
            sent = await bot.answer_guest_query(query_id, _article(text, parse_mode=parse_mode))
            return sent.inline_message_id
        except TelegramError as e:
            log.warning("answer_guest_query failed: %s", e)
            return None

    if not svc.settings.guest:
        await answer("Guest mode is disabled for this bot.")
        return
    res = await check_access(svc, bot, msg.from_user, source="guest", chat_id=msg.chat_id,
                             chat_type=msg.chat.type if msg.chat else None)
    if res is None:
        return
    if not res.role.can_transcribe:
        if res.role != Role.BLOCKED:
            await answer("🔒 You're not allowed to use this bot.")
        return
    media = extract_media(msg.reply_to_message)
    if media is None:
        await answer(f"Reply to a voice message and mention @{svc.bot_username} to transcribe it.")
        return

    inline_id = await answer("🎧 Transcribing…")
    try:
        result = await transcribe_media(svc, bot, media, access=res, chat_id=msg.chat_id,
                                        chat_type=msg.chat.type if msg.chat else None, source="guest")
    except UserFacingError as e:
        await _finish(bot, inline_id, html.escape(str(e)), lambda t: answer(t, ParseMode.HTML))
        return
    except Exception:
        log.exception("guest transcription failed")
        await _finish(bot, inline_id, "⚠️ Transcription failed.", lambda t: answer(t, ParseMode.HTML))
        return

    chunks = fmt.render(result)
    text = "\n\n".join(chunks) if chunks else ""
    if not chunks or len(text) > INLINE_MAX:
        first = chunks[0] if chunks else f"<i>{html.escape(fmt.header(result))}</i>"
        sent = await _send_private(bot, res.user.id, result)
        note = ("full text sent to you privately" if sent
                else f"open @{svc.bot_username} and press Start to receive full transcripts")
        text = first[: INLINE_MAX - 120] + f"\n\n<i>… (truncated; {note})</i>"
    await _finish(bot, inline_id, text, lambda t: answer(t, ParseMode.HTML))


async def _finish(bot, inline_id: str | None, text: str, fallback) -> None:
    if inline_id:
        try:
            await bot.edit_message_text(text, inline_message_id=inline_id, parse_mode=ParseMode.HTML)
            return
        except TelegramError as e:
            log.warning("editing guest message failed: %s", e)
    await fallback(text)


async def _send_private(bot, user_id: int, result) -> bool:
    try:
        await deliver(bot, result, chat_id=user_id)
        return True
    except TelegramError as e:
        log.info("could not DM user %s the full transcript: %s", user_id, e)
        return False

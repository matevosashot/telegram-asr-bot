"""Telegram Business connected-bot mode.

When you connect the bot to your account (Settings → Telegram Business → Chatbots), it
sees messages in the private chats you select. Reply to a voice message there with "/t"
(or "т", "տ") and the transcript is sent to you privately (or posted into the chat when
``business_reply_in_chat`` is on and the bot has the "reply" right).
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from .services import services
from .transcribe import UserFacingError, check_access, deliver, extract_media, transcribe_media

log = logging.getLogger(__name__)

TRIGGERS = {"/t", "t", "т", "տ", "/т", "/տ"}


async def business_connection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    conn = update.business_connection
    if conn is None:
        return
    can_reply = bool(conn.rights and conn.rights.can_reply)
    await svc.db.upsert_business_connection(conn.id, conn.user.id, conn.user_chat_id, can_reply, conn.is_enabled)
    res = await check_access(svc, context.bot, conn.user, source="business")
    if res is None or not conn.is_enabled:
        return
    if res.role.can_transcribe:
        text = ("🔗 Business connection active. In a connected chat, reply to a voice message with /t "
                "and I'll send you the transcript here.")
        if not can_reply and svc.settings.business_reply_in_chat:
            text += "\n(The bot has no 'reply' right, so transcripts will be sent here, not into the chat.)"
        try:
            await context.bot.send_message(conn.user_chat_id, text)
        except TelegramError as e:
            log.info("could not greet business user: %s", e)


async def business_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    msg = update.business_message
    if msg is None or not svc.settings.business or not msg.business_connection_id:
        return
    conn = await svc.db.get_business_connection(msg.business_connection_id)
    if conn is None:
        try:
            bc = await context.bot.get_business_connection(msg.business_connection_id)
        except TelegramError as e:
            log.warning("unknown business connection %s: %s", msg.business_connection_id, e)
            return
        await svc.db.upsert_business_connection(bc.id, bc.user.id, bc.user_chat_id,
                                                bool(bc.rights and bc.rights.can_reply), bc.is_enabled)
        conn = await svc.db.get_business_connection(bc.id)
    if not conn or not conn.is_enabled:
        return

    from_owner = msg.from_user is not None and msg.from_user.id == conn.user_id
    text = (msg.text or "").strip().lower()
    if from_owner and text in TRIGGERS:
        target = msg.reply_to_message
    elif not from_owner and svc.settings.business_auto_transcribe and extract_media(msg):
        target = msg
    else:
        return
    media = extract_media(target)
    if media is None:
        return

    # The connected account's owner is the one using the bot, whoever sent the voice.
    if from_owner:
        res = await check_access(svc, context.bot, msg.from_user, source="business", chat_id=msg.chat_id,
                                 chat_type=msg.chat.type)
    else:
        res = await svc.access.lookup(conn.user_id)
    if res is None or not res.role.can_transcribe:
        return

    in_chat = svc.settings.business_reply_in_chat and conn.can_reply and from_owner
    try:
        result = await transcribe_media(svc, context.bot, media, access=res, chat_id=msg.chat_id,
                                        chat_type=msg.chat.type, source="business")
    except UserFacingError as e:
        await _safe(context.bot.send_message(conn.user_chat_id, str(e)))
        return
    except Exception:
        log.exception("business transcription failed")
        await _safe(context.bot.send_message(conn.user_chat_id, "⚠️ Transcription failed."))
        return
    if in_chat:
        try:
            await deliver(context.bot, result, chat_id=msg.chat_id, reply_to=target.message_id,
                          business_connection_id=conn.id)
            return
        except TelegramError as e:
            log.warning("replying in business chat failed, sending privately: %s", e)
    who = msg.chat.full_name or msg.chat.username or str(msg.chat_id)
    await _safe(context.bot.send_message(conn.user_chat_id, f"🗨 Voice from chat with {who}:"))
    await deliver(context.bot, result, chat_id=conn.user_chat_id)


async def _safe(coro) -> None:
    try:
        await coro
    except TelegramError as e:
        log.warning("send failed: %s", e)

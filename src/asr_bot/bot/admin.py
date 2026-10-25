"""Owner-only commands and the /admin inline menu."""

from __future__ import annotations

import contextlib
import html
import time
from datetime import datetime

from telegram import (
    InlineKeyboardButton as B,
    InlineKeyboardMarkup,
    KeyboardButton,
    KeyboardButtonRequestUsers,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from .. import languages
from ..auth import parse_target
from ..db import UserRow
from ..engine import ENGINE_SCORING, NO_DETECTOR
from ..limits import fmt_duration
from .services import Services, services

PAGE = 8
LANGUAGE_PRESETS = ["hy", "ru", "en", "uk", "ka", "fa", "tr", "de", "fr", "es", "ar", "zh"]
PER_HOUR = [3, 5, 10, 20, 60, 0]
PER_DAY = [10, 20, 50, 100, 500, 0]
MAX_AUDIO = [60, 180, 300, 600, 1800, 0]
CONFIDENCE = [0.3, 0.5, 0.7, 0.9]
FEATURES = {
    "groups": "Groups: /t reply",
    "guest": "Guest Mode (@mention in any chat)",
    "business": "Business chats",
    "business_reply_in_chat": "Business: reply inside the chat",
    "business_auto_transcribe": "Business: auto-transcribe incoming voices",
    "notify_unknown_users": "Notify me about new users",
}
STATUS_ICON = {"allowed": "✅", "blocked": "⛔", "unknown": "❔"}
REQUEST_USERS_ID = 1


def _is_owner(update: Update, svc: Services) -> bool:
    return update.effective_user is not None and update.effective_user.id == svc.owner_id


def _inf(n: float, f=str) -> str:
    return "∞" if not n else f(n)


# ---- menu rendering --------------------------------------------------------------------


def main_menu(svc: Services) -> tuple[str, InlineKeyboardMarkup]:
    s = svc.settings
    cands = ", ".join(s.candidate_languages) or "any"
    loaded = svc.engines.loaded()
    text = (
        "⚙️ <b>Settings</b>\n\n"
        f"Engine: <b>{s.default_engine}</b>\n"
        f"Language detector: <b>{s.language_detector}</b>\n"
        f"Candidate languages: <b>{cands}</b>\n"
        f"Min. language confidence: <b>{s.min_language_prob:.0%}</b>\n"
        f"Limits: <b>{_inf(s.per_hour)}/hour · {_inf(s.per_day)}/day · "
        f"{_inf(s.max_audio_seconds, fmt_duration)} per file</b>\n"
        f"Loaded models: {', '.join(loaded) or 'none'}"
    )
    if svc.engines.unavailable:
        text += "\n\n<i>Unavailable: " + html.escape(", ".join(svc.engines.unavailable)) + "</i>"
    kb = InlineKeyboardMarkup([
        [B("🎙 Engine", callback_data="adm:m:engine"), B("🌐 Detector", callback_data="adm:m:detector")],
        [B("🗣 Languages", callback_data="adm:m:langs"), B("🎯 Confidence", callback_data="adm:m:conf")],
        [B("⏱ Limits", callback_data="adm:m:limits"), B("🧩 Features", callback_data="adm:m:features")],
        [B("👥 Users", callback_data="adm:users:0"), B("📜 Recent", callback_data="adm:m:recent"),
         B("📊 Stats", callback_data="adm:m:stats")],
        [B("🧹 Unload models", callback_data="adm:unload"), B("✖ Close", callback_data="adm:close")],
    ])
    return text, kb


def _back(target: str = "adm:m:main") -> list[B]:
    return [B("« Back", callback_data=target)]


def _mark(on: bool) -> str:
    return "✅ " if on else ""


def engine_menu(svc: Services):
    cur = svc.settings.default_engine
    rows = [[B(_mark(cur == "auto") + "auto – route by language", callback_data="adm:set:default_engine:auto")]]
    for name, eng in svc.engines.engines.items():
        langs = "all" if eng.supported_languages is None else ",".join(sorted(eng.supported_languages))[:20]
        rows.append([B(_mark(cur == name) + f"{name} – {eng.display_name} ({langs})",
                       callback_data=f"adm:set:default_engine:{name}")])
    rules = ", ".join(f"{k}→{v}" for k, v in svc.config.route_rules.items())
    text = ("🎙 <b>Default engine</b>\n\n<b>auto</b> picks the engine from the detected language: "
            f"{html.escape(rules)}, otherwise the first of {', '.join(svc.config.route_fallback)} "
            "that supports it.\nUsers can override it for themselves with /engine.")
    return text, InlineKeyboardMarkup([*rows, _back()])


def detector_menu(svc: Services):
    cur = svc.settings.language_detector
    rows = [[B(_mark(cur == n) + f"{n} – {d.display_name}", callback_data=f"adm:set:language_detector:{n}")]
            for n, d in svc.engines.detectors.items()]
    rows.append([B(_mark(cur == ENGINE_SCORING) + "engine-scoring – try each candidate language",
                   callback_data=f"adm:set:language_detector:{ENGINE_SCORING}")])
    rows.append([B(_mark(cur == NO_DETECTOR) + "none – no detection, no language tag",
                   callback_data=f"adm:set:language_detector:{NO_DETECTOR}")])
    text = ("🌐 <b>Language detection</b>\n\n"
            "Detectors return a probability per language; only the candidate languages are kept. "
            "<b>none</b> skips detection: the engine decides, and replies carry no language tag.\n"
            "<b>engine-scoring</b> transcribes the first 15 s once per candidate language and keeps "
            "the most likely one (slower, needs candidate languages).")
    return text, InlineKeyboardMarkup([*rows, _back()])


def langs_menu(svc: Services):
    cur = list(svc.settings.candidate_languages)
    presets = list(dict.fromkeys([*cur, *LANGUAGE_PRESETS]))
    rows, row = [], []
    for code in presets:
        row.append(B(("✅ " if code in cur else "") + f"{languages.flag(code)} {code}",
                     callback_data=f"adm:lang:{code}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([B("Any language (clear)", callback_data="adm:lang:-")])
    text = ("🗣 <b>Candidate languages</b>\n\nLanguage detection only chooses among these "
            "(empty = any language). Tap to toggle. Any ISO code can be set with "
            "<code>/languages hy ru en</code>.\n\nCurrent: <b>"
            + (", ".join(f"{c} ({languages.name(c)})" for c in cur) or "any") + "</b>")
    return text, InlineKeyboardMarkup([*rows, _back()])


def conf_menu(svc: Services):
    cur = svc.settings.min_language_prob
    rows = [[B(_mark(abs(cur - c) < 1e-6) + f"{c:.0%}", callback_data=f"adm:set:min_language_prob:{c}")
             for c in CONFIDENCE]]
    text = ("🎯 <b>Minimum language confidence</b>\n\nIf the detector is less sure than this, the engine "
            "gets no language hint and decides itself.")
    return text, InlineKeyboardMarkup([*rows, _back()])


def limits_menu(svc: Services):
    s = svc.settings
    rows = [
        [B("Requests per hour:", callback_data="adm:noop")],
        [B(_mark(s.per_hour == v) + _inf(v), callback_data=f"adm:set:per_hour:{v}") for v in PER_HOUR],
        [B("Requests per day:", callback_data="adm:noop")],
        [B(_mark(s.per_day == v) + _inf(v), callback_data=f"adm:set:per_day:{v}") for v in PER_DAY],
        [B("Max audio per file:", callback_data="adm:noop")],
        [B(_mark(s.max_audio_seconds == v) + _inf(v, lambda x: f"{x // 60}m"),
           callback_data=f"adm:set:max_audio_seconds:{v}") for v in MAX_AUDIO],
        _back(),
    ]
    return "⏱ <b>Limits</b> for allowed users (you are never limited).", InlineKeyboardMarkup(rows)


def features_menu(svc: Services):
    rows = [[B(("🟢 " if getattr(svc.settings, k) else "⚪ ") + label, callback_data=f"adm:tog:{k}")]
            for k, label in FEATURES.items()]
    text = ("🧩 <b>Features</b>\n\nGuest Mode and Business mode must also be enabled for the bot in "
            "@BotFather (Bot Settings).")
    return text, InlineKeyboardMarkup([*rows, _back()])


async def users_menu(svc: Services, page: int):
    total = await svc.db.count_users()
    users = await svc.db.list_users(limit=PAGE, offset=page * PAGE)
    pending = await svc.db.list_pending_usernames()
    rows = [[B(f"{STATUS_ICON.get(u.status, '')} {u.display}"[:60], callback_data=f"adm:u:{u.id}")]
            for u in users if u.id != svc.owner_id]
    nav = []
    if page > 0:
        nav.append(B("‹ Prev", callback_data=f"adm:users:{page - 1}"))
    if (page + 1) * PAGE < total:
        nav.append(B("Next ›", callback_data=f"adm:users:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([B("📇 Pick from contacts", callback_data="adm:pick")])
    rows.append(_back())
    text = (f"👥 <b>Users</b> ({await svc.db.count_users('allowed')} allowed, {total} known)\n\n"
            "Tap a user to allow / remove / block. To add someone who hasn't written yet: "
            "<code>/allow @username</code> (they're approved when they first message the bot), "
            "or pick them from your contacts.")
    if pending:
        text += "\n\nWaiting for first contact: " + ", ".join("@" + html.escape(p) for p in pending)
    return text, InlineKeyboardMarkup(rows)


async def user_menu(svc: Services, user_id: int):
    u = await svc.db.get_user(user_id)
    if u is None:
        return "User not found.", InlineKeyboardMarkup([_back("adm:users:0")])
    recent = await svc.db.recent_interactions(5, user_id=user_id)
    lines = [f"{_ts(r.ts)} {r.source}/{r.kind}" + (f" {r.language} {r.engine}" if r.kind == "transcribe" else "")
             for r in recent]
    text = (f"{STATUS_ICON.get(u.status, '')} <b>{html.escape(u.display)}</b>\nid: <code>{u.id}</code>\n"
            f"status: {u.status}\nengine: {u.engine or 'default'}\n"
            f"first seen: {_ts(u.first_seen)}\nlast seen: {_ts(u.last_seen)}\n\n"
            + ("\n".join(lines) or "no activity"))
    actions = []
    if u.status != "allowed":
        actions.append(B("✅ Allow", callback_data=f"adm:ua:allow:{u.id}"))
    if u.status == "allowed":
        actions.append(B("➖ Remove", callback_data=f"adm:ua:deny:{u.id}"))
    if u.status != "blocked":
        actions.append(B("⛔ Block", callback_data=f"adm:ua:block:{u.id}"))
    else:
        actions.append(B("Unblock", callback_data=f"adm:ua:deny:{u.id}"))
    return text, InlineKeyboardMarkup([actions, _back("adm:users:0")])


async def recent_text(svc: Services, limit: int = 15) -> str:
    rows = await svc.db.recent_interactions(limit)
    if not rows:
        return "📜 No interactions yet."
    out = ["📜 <b>Recent activity</b>\n"]
    for r in rows:
        u = await svc.db.get_user(r.user_id) if r.user_id else None
        who = html.escape(u.display if u else str(r.user_id))
        extra = ""
        if r.kind == "transcribe":
            extra = f" · {r.language} · {r.engine} · {fmt_duration(r.duration or 0)}"
        elif r.detail:
            extra = " · " + html.escape(r.detail[:60])
        out.append(f"{_ts(r.ts)} {who}: {r.source}/{r.kind}{extra}")
    return "\n".join(out)


async def stats_text(svc: Services) -> str:
    now = time.time()
    parts = ["📊 <b>Stats</b>"]
    for label, since in (("24h", now - 86400), ("7 days", now - 7 * 86400)):
        st = await svc.db.stats(since)
        parts.append(
            f"\n<b>{label}</b>: {st['count']} transcriptions, {fmt_duration(st['audio_seconds'])} audio, "
            f"{st['users']} users, avg {st['avg_latency']:.1f}s\n"
            f"engines: {_kv(st['engines'])}\nlanguages: {_kv(st['languages'])}\n"
            f"other: {_kv({k: v for k, v in st['kinds'].items() if k != 'transcribe'})}")
    return "\n".join(parts)


def _kv(d: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items()) or "–"


def _ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


# ---- commands --------------------------------------------------------------------------


async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    text, kb = main_menu(svc)
    await update.effective_message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)


async def _target_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE, action: str) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    msg = update.effective_message
    targets = [parse_target(a) for a in context.args]
    if not context.args and msg.reply_to_message:
        src = msg.reply_to_message
        origin = getattr(src, "forward_origin", None)
        sender = getattr(origin, "sender_user", None) if origin else src.from_user
        if origin is not None and sender is None:
            await msg.reply_text("That person hides their account in forwards, so I can't see who they are. "
                                 f"Use /{action} @username, or pick them with /users → Pick from contacts.")
            return
        if sender and sender.id != svc.owner_id:
            await svc.db.ensure_user(sender.id, sender.username, sender.first_name, sender.last_name)
            targets = [sender.id]
    if not targets or any(t is None for t in targets):
        await msg.reply_text(f"Usage: /{action} @username or /{action} 123456789 "
                             "(or reply to a forwarded message from that person)")
        return
    lines = []
    for t in targets:
        if action == "allow":
            outcome, user = await svc.access.allow(t)
            if outcome == "pending":
                lines.append(f"🕓 {t}: not seen yet – will be allowed when they first message the bot.")
            else:
                lines.append(f"✅ {user.display if user else t} allowed.")
                await _greet(context, user)
        elif action == "deny":
            outcome, user = await svc.access.deny(t)
            lines.append({"removed": f"➖ {user.display if user else t} removed from the whitelist.",
                          "pending_removed": f"➖ Pending {t} removed.",
                          "not_found": f"❓ {t} not found."}[outcome])
        else:
            outcome, user = await svc.access.block(t)
            lines.append({"blocked": f"⛔ {user.display if user else t} blocked.",
                          "owner": "You can't block yourself.",
                          "not_found": f"❓ {t} not found (the bot hasn't seen this username)."}[outcome])
    await msg.reply_text("\n".join(lines))


async def _greet(context: ContextTypes.DEFAULT_TYPE, user: UserRow | None) -> None:
    if user is None:
        return
    with contextlib.suppress(TelegramError):
        await context.bot.send_message(user.id, "✅ You now have access. Send me a voice message to transcribe it. /help")


async def allow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _target_cmd(update, context, "allow")


async def deny_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _target_cmd(update, context, "deny")


async def block_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _target_cmd(update, context, "block")


async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    text, kb = await users_menu(svc, 0)
    await update.effective_message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)


async def recent_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    n = int(context.args[0]) if context.args and context.args[0].isdigit() else 15
    await update.effective_message.reply_text(await recent_text(svc, min(n, 50)), parse_mode=ParseMode.HTML)


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    await update.effective_message.reply_text(await stats_text(svc), parse_mode=ParseMode.HTML)


async def languages_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    msg = update.effective_message
    if not context.args:
        text, kb = langs_menu(svc)
        await msg.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
        return
    codes = [] if context.args[0] in ("any", "-", "all") else [
        languages.normalize(a.strip(",")) for a in context.args]
    unknown = [c for c in codes if c not in languages.LANGUAGES]
    if unknown:
        await msg.reply_text(f"Unknown language code(s): {', '.join(map(str, unknown))}")
        return
    await svc.settings.set("candidate_languages", list(dict.fromkeys(codes)))
    await msg.reply_text("Candidate languages: " + (", ".join(codes) or "any"))


async def unload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    if not _is_owner(update, svc):
        return
    unloaded = await svc.engines.call(svc.engines.unload_all)
    text = "Unloaded: " + (", ".join(unloaded) or "nothing was loaded")
    kept = [n for n in svc.engines.loaded() if svc.engines.is_pinned(n)]
    if kept:
        text += "\nKept loaded (pinned): " + ", ".join(kept)
    await update.effective_message.reply_text(text)


async def users_shared(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The owner picked contacts with the 'Pick from contacts' keyboard button."""
    svc = services(context)
    if not _is_owner(update, svc):
        return
    shared = update.effective_message.users_shared
    lines = []
    for su in shared.users:
        await svc.db.ensure_user(su.user_id, su.username, su.first_name, su.last_name)
        await svc.db.set_status(su.user_id, "allowed")
        user = await svc.db.get_user(su.user_id)
        lines.append(f"✅ {user.display if user else su.user_id} allowed.")
        await _greet(context, user)
    await update.effective_message.reply_text("\n".join(lines) or "Nobody selected.",
                                              reply_markup=ReplyKeyboardRemove())


# ---- callbacks -------------------------------------------------------------------------


async def access_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Buttons on the 'new user tried the bot' notification."""
    svc = services(context)
    q = update.callback_query
    if not _is_owner(update, svc):
        await q.answer()
        return
    _, action, uid = q.data.split(":")
    user_id = int(uid)
    if action == "allow":
        await svc.access.allow(user_id)
        user = await svc.db.get_user(user_id)
        await _greet(context, user)
        note = "✅ Allowed"
    elif action == "block":
        await svc.access.block(user_id)
        note = "⛔ Blocked"
    else:
        note = "🙈 Ignored"
    await q.answer(note)
    original = getattr(q.message, "text_html", None) or "👤 New user"
    with contextlib.suppress(TelegramError):
        await q.edit_message_text(original + f"\n\n<b>{note}</b>", parse_mode=ParseMode.HTML)


async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc = services(context)
    q = update.callback_query
    if not _is_owner(update, svc):
        await q.answer("Owner only", show_alert=True)
        return
    parts = q.data.split(":")
    cmd = parts[1]
    s = svc.settings
    view = None
    if cmd == "noop":
        await q.answer()
        return
    if cmd == "close":
        await q.answer()
        with contextlib.suppress(TelegramError):
            await q.message.delete()
        return
    if cmd == "m":
        view = parts[2]
    elif cmd == "set":
        key, raw = parts[2], ":".join(parts[3:])
        current = getattr(s, key)
        value = type(current)(float(raw)) if isinstance(current, (int, float)) and not isinstance(current, bool) else raw
        if key == "default_engine" and value != "auto" and value not in svc.engines.engines:
            await q.answer("Engine not available", show_alert=True)
            return
        await s.set(key, value)
        view = {"default_engine": "engine", "language_detector": "detector",
                "min_language_prob": "conf"}.get(key, "limits")
    elif cmd == "tog":
        key = parts[2]
        await s.set(key, not getattr(s, key))
        view = "features"
    elif cmd == "lang":
        code = parts[2]
        cur = list(s.candidate_languages)
        if code == "-":
            cur = []
        elif code in cur:
            cur.remove(code)
        else:
            cur.append(code)
        await s.set("candidate_languages", cur)
        view = "langs"
    elif cmd == "unload":
        # Answer first: unloading waits for queued transcriptions on the GPU thread.
        await q.answer("Unloading models…")
        await svc.engines.call(svc.engines.unload_all)
        view = "main"
    elif cmd == "users":
        text, kb = await users_menu(svc, int(parts[2]))
        await _show(q, text, kb)
        return
    elif cmd == "u":
        text, kb = await user_menu(svc, int(parts[2]))
        await _show(q, text, kb)
        return
    elif cmd == "ua":
        action, uid = parts[2], int(parts[3])
        if action == "allow":
            await svc.access.allow(uid)
            await _greet(context, await svc.db.get_user(uid))
        elif action == "deny":
            await svc.access.deny(uid)
        elif action == "block":
            await svc.access.block(uid)
        text, kb = await user_menu(svc, uid)
        await _show(q, text, kb)
        return
    elif cmd == "pick":
        await q.answer()
        kb = ReplyKeyboardMarkup(
            [[KeyboardButton("📇 Choose people to allow", request_users=KeyboardButtonRequestUsers(
                request_id=REQUEST_USERS_ID, user_is_bot=False, max_quantity=10,
                request_name=True, request_username=True))]],
            one_time_keyboard=True, resize_keyboard=True)
        await q.message.reply_text("Tap the button below to choose contacts.", reply_markup=kb)
        return
    renderers = {"main": main_menu, "engine": engine_menu, "detector": detector_menu, "langs": langs_menu,
                 "conf": conf_menu, "limits": limits_menu, "features": features_menu}
    if view in renderers:
        text, kb = renderers[view](svc)
    elif view == "recent":
        text, kb = await recent_text(svc), InlineKeyboardMarkup([_back()])
    elif view == "stats":
        text, kb = await stats_text(svc), InlineKeyboardMarkup([_back()])
    else:
        await q.answer("?")
        return
    await _show(q, text, kb)


async def _show(q, text: str, kb: InlineKeyboardMarkup) -> None:
    with contextlib.suppress(TelegramError):
        await q.answer()
    try:
        await q.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise

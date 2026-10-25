"""Bot handlers driven through Application.process_update with a fake Telegram transport."""

from __future__ import annotations

from .conftest import OWNER_ID, message_dict, send_update, voice_dict


def _svc(app):
    from asr_bot.bot.services import KEY

    return app.bot_data[KEY]


async def test_owner_voice_is_transcribed(app, fake_tg):
    await send_update(app, {"message": message_dict(OWNER_ID, voice=voice_dict())})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))
    assert fake_tg.find("getFile")
    final = [p for p in fake_tg.find("editMessageText") if "բարեւ" in p["text"]][0]
    assert "Armenian" in final["text"]
    rows = await _svc(app).db.recent_interactions(5)
    assert rows[0].kind == "transcribe" and rows[0].engine == "armo"


async def test_unknown_user_is_refused_and_owner_notified_once(app, fake_tg):
    for _ in range(2):
        await send_update(app, {"message": message_dict(55, voice=voice_dict(), username="stranger")})
    await fake_tg.wait_for(lambda t: len(t.find("sendMessage")) >= 3)
    to_owner = [p for p in fake_tg.find("sendMessage") if int(p["chat_id"]) == OWNER_ID]
    assert len(to_owner) == 1 and "stranger" in to_owner[0]["text"]
    assert "acc:allow:55" in str(to_owner[0]["reply_markup"])
    assert not fake_tg.find("getFile")


async def test_owner_allows_via_button_then_user_can_transcribe(app, fake_tg):
    await send_update(app, {"message": message_dict(56, text="hi", username="pal")})
    await send_update(app, {"callback_query": {
        "id": "cb1", "chat_instance": "x", "data": "acc:allow:56", "from": {"id": OWNER_ID, "is_bot": False,
                                                                           "first_name": "Owner"},
        "message": message_dict(OWNER_ID, text="👤 New user", message_id=77)}})
    await fake_tg.wait_for(lambda t: any("now have access" in x for x in t.texts()))
    fake_tg.calls.clear()
    await send_update(app, {"message": message_dict(56, voice=voice_dict(), username="pal")})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))


async def test_allow_by_username_before_first_contact(app, fake_tg):
    await send_update(app, {"message": message_dict(OWNER_ID, text="/allow @futurefriend")})
    await fake_tg.wait_for(lambda t: any("not seen yet" in x for x in t.texts()))
    await send_update(app, {"message": message_dict(57, voice=voice_dict(), username="FutureFriend")})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))
    await fake_tg.wait_for(lambda t: any("pre-authorized" in x for x in t.texts()))


async def test_non_owner_cannot_use_admin(app, fake_tg):
    await _svc(app).access.allow(58)
    await send_update(app, {"message": message_dict(58, text="/allow @x")})
    await send_update(app, {"message": message_dict(58, text="/admin")})
    await send_update(app, {"callback_query": {
        "id": "cb2", "chat_instance": "x", "data": "adm:set:per_hour:100",
        "from": {"id": 58, "is_bot": False, "first_name": "U"}, "message": message_dict(58, text="m")}})
    await fake_tg.wait_for(lambda t: t.find("answerCallbackQuery"))
    assert _svc(app).settings.per_hour == 2
    assert not await _svc(app).db.list_pending_usernames()


async def test_rate_limit_message(app, fake_tg):
    await _svc(app).access.allow(59)
    for i in range(3):
        await send_update(app, {"message": message_dict(59, voice=voice_dict(), message_id=20 + i)})
        await fake_tg.wait_for(lambda t, n=i: sum("բարեւ" in x or "Limit" in x for x in t.texts()) > n)
    assert any("Limit reached: 2 requests per hour" in x for x in fake_tg.texts())


async def test_too_long_audio_rejected_before_download(app, fake_tg):
    await _svc(app).access.allow(60)
    await send_update(app, {"message": message_dict(60, voice=voice_dict(duration=600))})
    await fake_tg.wait_for(lambda t: any("too long" in x for x in t.texts()))
    assert not fake_tg.find("getFile")


async def test_owner_has_no_limits(app, fake_tg):
    await send_update(app, {"message": message_dict(OWNER_ID, voice=voice_dict(duration=600))})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))


async def test_group_t_command_replying_to_voice(app, fake_tg):
    voice_msg = message_dict(61, chat_id=-500, chat_type="supergroup", voice=voice_dict(), message_id=5)
    cmd = message_dict(OWNER_ID, chat_id=-500, chat_type="supergroup", text="/t", reply_to=voice_msg,
                       message_id=6)
    await send_update(app, {"message": cmd})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))
    placeholder = fake_tg.find("sendMessage")[0]
    assert int(placeholder["chat_id"]) == -500
    assert '"message_id": 5' in str(placeholder.get("reply_parameters")).replace("'", '"')


async def test_group_t_without_reply(app, fake_tg):
    await send_update(app, {"message": message_dict(OWNER_ID, chat_id=-500, chat_type="supergroup", text="/t")})
    await fake_tg.wait_for(lambda t: any("Reply to a voice" in x for x in t.texts()))


async def test_guest_mode(app, fake_tg):
    voice_msg = message_dict(62, chat_id=62, voice=voice_dict(), message_id=5)
    guest = message_dict(OWNER_ID, chat_id=62, text="@testbot", reply_to=voice_msg, message_id=6,
                         guest_query_id="gq1")
    await send_update(app, {"guest_message": guest})
    await fake_tg.wait_for(lambda t: any("բարեւ" in p.get("text", "") for p in t.find("editMessageText")))
    assert fake_tg.find("answerGuestQuery")[0]["guest_query_id"] == "gq1"
    edit = fake_tg.find("editMessageText")[-1]
    assert edit["inline_message_id"] == "inline-1"


async def test_guest_mode_unauthorized(app, fake_tg):
    voice_msg = message_dict(62, chat_id=62, voice=voice_dict(), message_id=5)
    guest = message_dict(63, chat_id=62, text="@testbot", reply_to=voice_msg, message_id=6, guest_query_id="gq2")
    await send_update(app, {"guest_message": guest})
    await fake_tg.wait_for(lambda t: t.find("answerGuestQuery"))
    assert "not allowed" in str(fake_tg.find("answerGuestQuery")[0]["result"])
    assert not fake_tg.find("getFile")


async def test_guest_edit_failure_falls_back_to_second_answer(app, fake_tg):
    fake_tg.fail_methods.add("editMessageText")
    voice_msg = message_dict(62, chat_id=62, voice=voice_dict(), message_id=5)
    guest = message_dict(OWNER_ID, chat_id=62, text="@testbot", reply_to=voice_msg, message_id=6,
                         guest_query_id="gq3")
    await send_update(app, {"guest_message": guest})
    await fake_tg.wait_for(lambda t: len(t.find("answerGuestQuery")) == 2)
    assert "բարեւ" in str(fake_tg.find("answerGuestQuery")[1]["result"])


async def test_business_trigger_sends_transcript_privately(app, fake_tg):
    conn = {"id": "bc1", "user": {"id": OWNER_ID, "is_bot": False, "first_name": "Owner"},
            "user_chat_id": OWNER_ID, "date": 1, "is_enabled": True, "rights": {"can_reply": True}}
    await send_update(app, {"business_connection": conn})
    voice_msg = message_dict(64, chat_id=64, voice=voice_dict(), message_id=5)
    trigger = message_dict(OWNER_ID, chat_id=64, text="/t", reply_to=voice_msg, message_id=6,
                           business_connection_id="bc1")
    fake_tg.calls.clear()
    await send_update(app, {"business_message": trigger})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))
    final = [p for p in fake_tg.find("sendMessage") if "բարեւ" in p["text"]][0]
    assert int(final["chat_id"]) == OWNER_ID and "business_connection_id" not in final


async def test_business_reply_in_chat_setting(app, fake_tg):
    await _svc(app).settings.set("business_reply_in_chat", True)
    conn = {"id": "bc2", "user": {"id": OWNER_ID, "is_bot": False, "first_name": "Owner"},
            "user_chat_id": OWNER_ID, "date": 1, "is_enabled": True, "rights": {"can_reply": True}}
    await send_update(app, {"business_connection": conn})
    voice_msg = message_dict(65, chat_id=65, voice=voice_dict(), message_id=5)
    trigger = message_dict(OWNER_ID, chat_id=65, text="t", reply_to=voice_msg, message_id=6,
                           business_connection_id="bc2")
    fake_tg.calls.clear()
    await send_update(app, {"business_message": trigger})
    await fake_tg.wait_for(lambda t: any("բարեւ" in x for x in t.texts()))
    final = [p for p in fake_tg.find("sendMessage") if "բարեւ" in p["text"]][0]
    assert int(final["chat_id"]) == 65 and final["business_connection_id"] == "bc2"


async def test_admin_menu_changes_settings(app, fake_tg):
    owner = {"id": OWNER_ID, "is_bot": False, "first_name": "Owner"}
    menu_msg = message_dict(OWNER_ID, text="menu", message_id=90)
    for i, data in enumerate(["adm:set:default_engine:ru", "adm:set:per_hour:20", "adm:lang:ru",
                              "adm:tog:guest", "adm:set:language_detector:engine-scoring", "adm:m:main"]):
        await send_update(app, {"callback_query": {"id": f"c{i}", "chat_instance": "x", "data": data,
                                                   "from": owner, "message": menu_msg}})
    s = _svc(app).settings
    assert s.default_engine == "ru" and s.per_hour == 20 and s.candidate_languages == ["hy", "en"]
    assert s.guest is False and s.language_detector == "engine-scoring"
    assert "Settings" in fake_tg.find("editMessageText")[-1]["text"]


async def test_admin_users_and_stats_views(app, fake_tg):
    owner = {"id": OWNER_ID, "is_bot": False, "first_name": "Owner"}
    await _svc(app).access.check(70, "someone", "Some", "One")
    menu_msg = message_dict(OWNER_ID, text="menu", message_id=91)
    for i, data in enumerate(["adm:users:0", "adm:u:70", "adm:ua:allow:70", "adm:m:stats", "adm:m:recent"]):
        await send_update(app, {"callback_query": {"id": f"u{i}", "chat_instance": "x", "data": data,
                                                   "from": owner, "message": menu_msg}})
    texts = [p["text"] for p in fake_tg.find("editMessageText")]
    assert any("Users" in t for t in texts) and any("Stats" in t for t in texts)
    assert (await _svc(app).db.get_user(70)).status == "allowed"


async def test_users_shared_allows_contacts(app, fake_tg):
    msg = message_dict(OWNER_ID, users_shared={"request_id": 1, "users": [{"user_id": 71, "username": "c1",
                                                                            "first_name": "C"}]})
    await send_update(app, {"message": msg})
    await fake_tg.wait_for(lambda t: any("allowed" in x for x in t.texts()))
    assert (await _svc(app).db.get_user(71)).status == "allowed"


async def test_user_engine_override(app, fake_tg):
    await _svc(app).access.allow(72)
    await send_update(app, {"callback_query": {"id": "e1", "chat_instance": "x", "data": "usr:eng:general",
                                               "from": {"id": 72, "is_bot": False, "first_name": "U"},
                                               "message": message_dict(72, text="m")}})
    await fake_tg.wait_for(lambda t: t.find("answerCallbackQuery"))
    await send_update(app, {"message": message_dict(72, voice=voice_dict())})
    await fake_tg.wait_for(lambda t: any("hello from general" in x for x in t.texts()))


async def test_concurrent_requests_cannot_bypass_limit(app, fake_tg):
    import asyncio

    await _svc(app).access.allow(73)
    await asyncio.gather(*(send_update(app, {"message": message_dict(73, voice=voice_dict(), message_id=30 + i)})
                           for i in range(5)))
    await fake_tg.wait_for(lambda t: sum("բարեւ" in x or "Limit" in x for x in t.texts()) >= 5)
    assert sum("բարեւ" in x for x in fake_tg.texts()) == 2  # per_hour = 2 in the test config


async def test_unknown_user_name_with_html_is_escaped(app, fake_tg):
    msg = message_dict(74, text="hi")
    msg["from"]["first_name"] = "A<b>B"
    await send_update(app, {"message": msg})
    await fake_tg.wait_for(lambda t: any(int(p["chat_id"]) == OWNER_ID for p in t.find("sendMessage")))
    note = [p for p in fake_tg.find("sendMessage") if int(p["chat_id"]) == OWNER_ID][0]
    assert "A&lt;b&gt;B" in note["text"]


async def test_allow_reply_to_hidden_forward(app, fake_tg):
    hidden = message_dict(OWNER_ID, text="fwd", message_id=40,
                          forward_origin={"type": "hidden_user", "date": 1, "sender_user_name": "Secret"})
    await send_update(app, {"message": message_dict(OWNER_ID, text="/allow", reply_to=hidden, message_id=41)})
    await fake_tg.wait_for(lambda t: any("hides their account" in x for x in t.texts()))

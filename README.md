<p align="center"><img src="assets/logo.svg" width="128" alt="asr-bot logo"></p>

# asr-bot

A Telegram bot that transcribes voice messages. It runs local, swappable ASR engines
(Meta Omnilingual ASR, OpenAI Whisper, Qwen3-ASR, Voxtral, plus your own) and detects the spoken
language automatically. Armenian is a first-class language.

* Send or **forward** a voice message, audio file (mp3/m4a/ogg/wav/…), round video, or video
  to the bot, and it replies with the transcript.
* **Groups**: add the bot, then reply to a voice message with `/t`.
* **Any chat, including 1:1 chats with friends** (Telegram *Guest Mode*): reply to a voice
  message and mention `@yourbot`. The transcript appears in that chat. No Premium needed.
* **Telegram Business** (optional): connect the bot to your account and reply `/t` to a
  voice in a connected chat. You get the transcript privately (or in the chat if you set it up that way).
* Owner-only **admin menu** (`/admin`):
  - default engine
  - language detector
  - candidate languages
  - limits
  - feature toggles
  - users
  - recent activity
  - stats
  - unloading models
* **Access control**: one owner (from the config file), plus a whitelist.
  - When a stranger writes to the bot, you get a DM with Allow / Block buttons.
  - `/allow @username` also works for people who haven't written yet; they're approved on first contact.
  - You can also pick people from your contacts.
* **Rate limits** for allowed users: requests per hour, requests per day, and max audio length.
  The owner is never limited.

## Install

```bash
conda create -n asrbot python=3.11 -y && conda activate asrbot
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"          # or pick extras: whisper, qwen, omni, voxtral, voxlingua
```

* ffmpeg/ffprobe must be on `PATH`.
* `omnilingual-asr` (fairseq2) pins `torch==2.8.0`. That's why torch is installed first.
* `pyarrow` is pinned below 21 because fairseq2 needs numpy 1.x.
* All extras coexist in one environment.

## Configure and run

```bash
asr-bot init-config --owner-id <your numeric Telegram id>   # writes ~/.config/asr-bot/config.toml
export TELEGRAM_BOT_TOKEN=123456:ABC...
asr-bot check        # token, ffmpeg, which engines/detectors are available
asr-bot run          # long polling
```

* **Your numeric id:** send `/whoami` to the bot, ask @userinfobot, or check the log after
  sending `/start`.
* **Config file:** see [`config.example.toml`](config.example.toml).
  - Everything under `[transcription]`, `[limits]` and `[features]` is a default; you can
    change it live from `/admin`, and those values are stored in the database.
  - The database lives in `~/.local/share/asr-bot/bot.db`.
* **Running as a service:** see [`deploy/asr-bot.service`](deploy/asr-bot.service) (systemd user unit).
* **Keeping models on the GPU:** `transcription.keep_loaded = ["omnilingual", "whisper"]` loads those
  engines (or detectors) at startup and never unloads them: not on idle, not on `/unload`, not when
  another model runs out of memory. Combine with `idle_unload_minutes = 0`. Check that they fit in VRAM together.
* **No language detection:** `language_detector = "none"` skips detection. Use it with a fixed
  `default_engine` (e.g. `omnilingual`); replies then have no language flag or name.

### BotFather settings

| What | Where in @BotFather | Needed for |
|---|---|---|
| Guest Mode | *Bot Settings → Guest Mode* (Mini App settings) | `@yourbot` replies in any chat |
| Business Mode | *Bot Settings → Business Mode* | Connected business chats |
| Allow groups | `/setjoingroups` | `/t` in groups |

Privacy mode can stay **on**: commands such as `/t` are still delivered in groups.

## Commands

* **Everyone allowed:**
  - `/start`, `/help`
  - `/t`: reply to a voice message with it.
  - `/engine`: personal engine choice.
  - `/usage`
  - `/whoami`
* **Owner** (private chat with the bot):

| Command | What it does |
|---|---|
| `/admin` | Settings and users menu (inline buttons) |
| `/allow @username\|id` | Whitelist someone. Unknown usernames are approved on first contact. Replying `/allow` to a forwarded message works too. |
| `/deny @username\|id` | Remove from the whitelist |
| `/block @username\|id` | Ignore completely |
| `/users`, `/recent [n]`, `/stats` | People, the activity log, usage numbers |
| `/languages hy ru en` | Candidate languages (`/languages any` to clear) |
| `/unload` | Free GPU memory now (except `keep_loaded` models). Models also unload after `idle_unload_minutes`. |

## How language detection and routing work

1. **Language ID** runs on the first 30 s of audio, separately from the ASR engine. The
   detector is chosen in `/admin`:
   * `whisper`: Whisper large-v3-turbo language-token probabilities (default).
   * `voxlingua`: SpeechBrain VoxLingua107 ECAPA (107 languages).
   * `engine-scoring`: transcribes the first 15 s once per candidate language with a
     scoring-capable engine (Whisper) and keeps the language with the best mean log-probability.
2. **Candidate languages** (default `hy, ru, en`): the detector's probabilities are limited
   to these and renormalised. This stops Armenian being confused with Georgian or Persian.
   An empty list means any language.
3. **Engine `auto`** routes by the detected language. Rules come first
   (`hy → omnilingual`), then the first engine in `routing.fallback` that supports the
   language (`qwen3` for its 30 languages, otherwise `whisper`).
4. The engine gets the language as a hint only when detection is confident
   (`min_language_prob`, default 50%) and the engine supports that language.

Each reply starts with a header line, for example: `🇦🇲 Armenian 99% · omnilingual · 0:42`.

## Engines

| Engine (`type`) | Languages | Notes |
|---|---|---|
| `omnilingual`: Meta Omnilingual ASR, `omniASR_LLM_Unlimited_3B_v2` | 1600+, incl. Armenian (`hye_Armn`) | Best Armenian; no punctuation or casing; the Unlimited models take any length; 16 GB download for 3B; a bf16 copy (8.8 GB) is cached in `~/.cache/asr-bot/weights` on first load, which cuts reloads from ~20 s to ~10 s (`fast_weights = false` disables it) |
| `faster-whisper`: Whisper `large-v3` | ~99 | Detects language; punctuation; weak on Armenian |
| `faster-whisper`: `Chillarmo/whisper-large-v3-turbo-hy-ct2` (as `whisper-hy`) | Armenian | Community fine-tune |
| `qwen3-asr`: `Qwen/Qwen3-ASR-1.7B` | 30 (no Armenian) | Strong for ru/en etc.; detects language |
| `voxtral`: Voxtral Mini 4B Realtime | 13 (no Armenian in the original) | Punctuation; fast; original or your own checkpoint, see below |

### Voxtral

[Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (en, fr, es, de,
ru, zh, ja, it, pt, nl, ar, hi, ko) runs on a small pure-PyTorch runtime in `src/asr_bot/asr/voxtral/`
(no `transformers` at run time; adapted from `voxtral-offline`, Apache-2.0). It is not used by `auto` routing
unless you add it to `[routing]`; pick it with `/engine` or in `/admin`.

```toml
[engines.voxtral]
type = "voxtral"
# checkpoint = "/path/to/checkpoint"   # unset = the original release
# delay_ms = 2400                      # 80..2400; the longest delay is the most accurate offline
# compile = false                      # torch.compile the decoder step (~50 s once, then cached)
# languages = ["en", "ru", "hy"]       # for a fine-tuned checkpoint
```

* **Default checkpoint:** the original `mistralai/Voxtral-Mini-4B-Realtime-2602`. The first use downloads
  `config.json`, `model.safetensors` and `tekken.json` (about 9 GB) to `~/.cache/asr-bot/voxtral/original`.
* **Custom checkpoint:** `checkpoint` is a local directory with `config.json`, a single `model.safetensors`
  and (optionally) `tekken.json`, in the original's format and architecture (for example a fine-tune).
  Without its own `tekken.json`, the original's tokenizer is used. Set `languages` to what the checkpoint
  handles: the bot routes and warns by that list (the model itself takes no language hint).
* **Conversion cache:** the first load converts the checkpoint once into a fused file
  (`~/.cache/asr-bot/voxtral`, ~9 GB per dtype); it is redone if the checkpoint changes.
  Later loads take about 6 s. Weights use about 9 GB of VRAM.

Measured on this machine (RTX 4090) with FLEURS test clips: see *Benchmarks* below.

### Adding your own engine

Implement `asr_bot.asr.ASRBackend` and register it, then add an `[engines.<name>]` table.

```python
# my_asr.py
from asr_bot.asr import ASRBackend, Transcript, register_backend

@register_backend("my-asr")
class MyASR(ASRBackend):
    display_name = "My ASR"
    supported_languages = {"hy"}        # None = any language
    detects_language = False

    def load(self):
        self.model = load_my_model(self.options["model"], device=self.device)

    def unload(self):
        self.model = None

    def is_loaded(self):
        return getattr(self, "model", None) is not None

    def transcribe(self, audio, language=None):
        # audio.samples: 16 kHz mono float32 numpy array; audio.ensure_path() gives a WAV path
        return Transcript(text=self.model(audio.samples), language=language or "hy")
```

* **Registering it:**
  - Put the module inside `asr_bot/asr/` and import it in `asr/base.py:_load_builtin_and_plugins`;
  - or ship it as a separate package with an entry point:
    ```toml
    [project.entry-points."asr_bot.backends"]
    my-asr = "my_asr:MyASR"
    ```
* **Config:**
  ```toml
  [engines.mine]
  type = "my-asr"
  model = "/path/to/weights"

  [routing]
  rules = { hy = "mine" }     # optional: make it the Armenian engine in auto mode
  ```
* **Language detectors** work the same way: implement `asr_bot.lid.LanguageDetector`
  (`probabilities(audio) -> {iso_code: p}`), registered through `asr_bot.detectors`.
* **Remote engines:** an engine that calls an HTTP service only has to implement `transcribe`.
  `load` and `unload` can be no-ops.

## Notes and limits

* **File size:** the cloud Bot API lets bots download files up to **20 MB**. For bigger files,
  run a [local Bot API server](https://github.com/tdlib/telegram-bot-api) and set
  `telegram.local_api_url`.
* **One job at a time:** GPU work runs on a single worker thread, and queued users see their
  position.
* **Privacy:** transcripts aren't stored by default (`storage.store_transcripts`). Only
  metadata is logged: who, when, duration, language, engine and latency.
* **Guest Mode replies:** a guest reply is one message, edited in place, limited to 4096
  characters. Longer transcripts are truncated there, and the full text is sent to you privately.

## Development

```bash
pip install -e ".[all,dev]"
pytest                       # unit + offline handler tests (fake Telegram transport)
python scripts/fetch_fleurs.py hy_am tests/data/cache/fleurs 6   # also ru_ru, en_us
pytest -m gpu -s             # real engines on FLEURS clips (downloads models)
ASR_BOT_TEST_TOKEN=... python scripts/live_smoke.py -c config.toml tests/data/cache/fleurs/hy_voice.ogg
```

The last command runs a live round trip through the real Bot API: it uploads the voice to the
owner's chat, then the bot downloads it, transcribes it and replies.

## Benchmarks

RTX 4090 with FLEURS test clips: 6 per language, 4–29 s each, several recorded very quietly.
CER is the character error rate after lower-casing and removing punctuation. RTF is processing
time divided by audio length, with the model already loaded. Lower is better for both. Reproduce with
`ASR_BOT_TEST_CLIPS=6 pytest -m gpu -s`.

| Engine | hy CER | ru CER | en CER | RTF (hy) |
|---|---|---|---|---|
| omnilingual (`omniASR_LLM_Unlimited_3B_v2`) | **1.9%** | 1.3% | 4.3% | 0.54 |
| whisper-hy (Armenian fine-tune, `max_chunk_seconds = 12`) | 6.1% | – | – | 0.09 |
| whisper (`large-v3`) | 16.3% | 0.3% | 1.3% | 0.27 |
| qwen3 (`Qwen3-ASR-1.7B`) | n/a | 0.4% | 1.2% | – |
| voxtral (original release) | n/a | 0.1% | 1.6% | 0.13 (ru/en) |

* **Language ID:** Whisper, VoxLingua107 and engine-scoring all identified hy, ru and en
  correctly on every test clip (candidates `hy, ru, en`).
* **Whisper on Armenian:** without `max_chunk_seconds`, Whisper stops mid-sentence on longer
  Armenian windows. Armenian needs several byte-level tokens per character, so the output token
  limit is reached quickly. Chunking at quiet points fixes this: whisper-hy CER goes from 15.2% to 6.1%.
* **Omnilingual output** has no punctuation or capitals and writes «եւ» rather than «և».
* **Default routing:** Armenian goes to Omnilingual (accuracy), Russian and English to Qwen3,
  and everything else to Whisper.
* **GPU memory (approx.):** Omnilingual 3B ~8.2 GB of weights (~11 GB in `nvidia-smi`), Voxtral ~9 GB, Qwen3 ~5 GB, each Whisper model ~2–4 GB.
  - If loading a model runs out of memory, the bot unloads the other models and retries.
  - Idle models unload after 15 min.

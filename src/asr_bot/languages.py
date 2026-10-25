"""Language code helpers. Internally the bot uses ISO-639-1 codes (as Whisper does)."""

from __future__ import annotations

# code: (English name, ISO-639-3, flag emoji)
LANGUAGES: dict[str, tuple[str, str, str]] = {
    "af": ("Afrikaans", "afr", "🇿🇦"), "am": ("Amharic", "amh", "🇪🇹"),
    "ar": ("Arabic", "arb", "🇸🇦"), "as": ("Assamese", "asm", "🇮🇳"),
    "az": ("Azerbaijani", "azj", "🇦🇿"), "ba": ("Bashkir", "bak", "🇷🇺"),
    "be": ("Belarusian", "bel", "🇧🇾"), "bg": ("Bulgarian", "bul", "🇧🇬"),
    "bn": ("Bengali", "ben", "🇧🇩"), "bo": ("Tibetan", "bod", "🏔"),
    "br": ("Breton", "bre", "🇫🇷"), "bs": ("Bosnian", "bos", "🇧🇦"),
    "ca": ("Catalan", "cat", "🇪🇸"), "cs": ("Czech", "ces", "🇨🇿"),
    "cy": ("Welsh", "cym", "🏴"), "da": ("Danish", "dan", "🇩🇰"),
    "de": ("German", "deu", "🇩🇪"), "el": ("Greek", "ell", "🇬🇷"),
    "en": ("English", "eng", "🇬🇧"), "es": ("Spanish", "spa", "🇪🇸"),
    "et": ("Estonian", "est", "🇪🇪"), "eu": ("Basque", "eus", "🇪🇸"),
    "fa": ("Persian", "pes", "🇮🇷"), "fi": ("Finnish", "fin", "🇫🇮"),
    "fil": ("Filipino", "fil", "🇵🇭"), "fo": ("Faroese", "fao", "🇫🇴"),
    "fr": ("French", "fra", "🇫🇷"), "gl": ("Galician", "glg", "🇪🇸"),
    "gu": ("Gujarati", "guj", "🇮🇳"), "ha": ("Hausa", "hau", "🇳🇬"),
    "haw": ("Hawaiian", "haw", "🇺🇸"), "he": ("Hebrew", "heb", "🇮🇱"),
    "hi": ("Hindi", "hin", "🇮🇳"), "hr": ("Croatian", "hrv", "🇭🇷"),
    "ht": ("Haitian Creole", "hat", "🇭🇹"), "hu": ("Hungarian", "hun", "🇭🇺"),
    "hy": ("Armenian", "hye", "🇦🇲"), "id": ("Indonesian", "ind", "🇮🇩"),
    "is": ("Icelandic", "isl", "🇮🇸"), "it": ("Italian", "ita", "🇮🇹"),
    "ja": ("Japanese", "jpn", "🇯🇵"), "jw": ("Javanese", "jav", "🇮🇩"),
    "ka": ("Georgian", "kat", "🇬🇪"), "kk": ("Kazakh", "kaz", "🇰🇿"),
    "km": ("Khmer", "khm", "🇰🇭"), "kn": ("Kannada", "kan", "🇮🇳"),
    "ko": ("Korean", "kor", "🇰🇷"), "la": ("Latin", "lat", "🏛"),
    "lb": ("Luxembourgish", "ltz", "🇱🇺"), "ln": ("Lingala", "lin", "🇨🇩"),
    "lo": ("Lao", "lao", "🇱🇦"), "lt": ("Lithuanian", "lit", "🇱🇹"),
    "lv": ("Latvian", "lvs", "🇱🇻"), "mg": ("Malagasy", "plt", "🇲🇬"),
    "mi": ("Maori", "mri", "🇳🇿"), "mk": ("Macedonian", "mkd", "🇲🇰"),
    "ml": ("Malayalam", "mal", "🇮🇳"), "mn": ("Mongolian", "khk", "🇲🇳"),
    "mr": ("Marathi", "mar", "🇮🇳"), "ms": ("Malay", "zsm", "🇲🇾"),
    "mt": ("Maltese", "mlt", "🇲🇹"), "my": ("Burmese", "mya", "🇲🇲"),
    "ne": ("Nepali", "npi", "🇳🇵"), "nl": ("Dutch", "nld", "🇳🇱"),
    "nn": ("Norwegian Nynorsk", "nno", "🇳🇴"), "no": ("Norwegian", "nob", "🇳🇴"),
    "oc": ("Occitan", "oci", "🇫🇷"), "pa": ("Punjabi", "pan", "🇮🇳"),
    "pl": ("Polish", "pol", "🇵🇱"), "ps": ("Pashto", "pbt", "🇦🇫"),
    "pt": ("Portuguese", "por", "🇵🇹"), "ro": ("Romanian", "ron", "🇷🇴"),
    "ru": ("Russian", "rus", "🇷🇺"), "sa": ("Sanskrit", "san", "🇮🇳"),
    "sd": ("Sindhi", "snd", "🇵🇰"), "si": ("Sinhala", "sin", "🇱🇰"),
    "sk": ("Slovak", "slk", "🇸🇰"), "sl": ("Slovenian", "slv", "🇸🇮"),
    "sn": ("Shona", "sna", "🇿🇼"), "so": ("Somali", "som", "🇸🇴"),
    "sq": ("Albanian", "als", "🇦🇱"), "sr": ("Serbian", "srp", "🇷🇸"),
    "su": ("Sundanese", "sun", "🇮🇩"), "sv": ("Swedish", "swe", "🇸🇪"),
    "sw": ("Swahili", "swh", "🇰🇪"), "ta": ("Tamil", "tam", "🇮🇳"),
    "te": ("Telugu", "tel", "🇮🇳"), "tg": ("Tajik", "tgk", "🇹🇯"),
    "th": ("Thai", "tha", "🇹🇭"), "tk": ("Turkmen", "tuk", "🇹🇲"),
    "tl": ("Tagalog", "tgl", "🇵🇭"), "tr": ("Turkish", "tur", "🇹🇷"),
    "tt": ("Tatar", "tat", "🇷🇺"), "uk": ("Ukrainian", "ukr", "🇺🇦"),
    "ur": ("Urdu", "urd", "🇵🇰"), "uz": ("Uzbek", "uzn", "🇺🇿"),
    "vi": ("Vietnamese", "vie", "🇻🇳"), "yi": ("Yiddish", "ydd", "🕍"),
    "yo": ("Yoruba", "yor", "🇳🇬"), "yue": ("Cantonese", "yue", "🇭🇰"),
    "zh": ("Chinese", "cmn", "🇨🇳"),
}

# Deprecated / alternative ISO-639-1 codes used by some models (e.g. VoxLingua107).
_ALIASES = {"iw": "he", "in": "id", "ji": "yi", "jv": "jw", "zh-cn": "zh", "nb": "no"}

_BY_NAME = {name.lower(): code for code, (name, _, _) in LANGUAGES.items()}
_BY_ISO3 = {iso3: code for code, (_, iso3, _) in LANGUAGES.items()}
# A few extra ISO-639-3 / macrolanguage codes seen in model outputs.
_BY_ISO3.update({"hyw": "hy", "zho": "zh", "ara": "ar", "fas": "fa", "msa": "ms", "nor": "no",
                 "lav": "lv", "est": "et", "uzb": "uz", "aze": "az", "sqi": "sq", "mon": "mn",
                 "nep": "ne", "pus": "ps", "swa": "sw", "yid": "yi", "jav": "jw", "kor": "ko"})


def name(code: str | None) -> str:
    if not code:
        return "unknown"
    return LANGUAGES.get(code, (code, "", ""))[0]


def flag(code: str | None) -> str:
    if not code:
        return "🏳"
    return LANGUAGES.get(code, ("", "", "🏳"))[2]


def to_iso3(code: str) -> str | None:
    entry = LANGUAGES.get(code)
    return entry[1] if entry else None


def normalize(code_or_name: str | None) -> str | None:
    """Turn 'hy', 'hye', 'hye_Armn', 'Armenian', 'hy: Armenian' into 'hy'."""
    if not code_or_name:
        return None
    s = code_or_name.strip()
    if ":" in s:  # VoxLingua labels look like "hy: Armenian"
        s = s.split(":", 1)[0].strip()
    low = s.lower()
    low = _ALIASES.get(low, low)
    if low in LANGUAGES:
        return low
    if low in _BY_NAME:
        return _BY_NAME[low]
    base = low.split("_", 1)[0].split("-", 1)[0]
    if base in LANGUAGES:
        return base
    if base in _BY_ISO3:
        return _BY_ISO3[base]
    return low


def restrict(probs: dict[str, float], candidates: list[str]) -> dict[str, float]:
    """Keep only candidate languages and renormalize. If none of the candidates got any
    probability mass, the original distribution is returned unchanged."""
    if not candidates:
        return probs
    kept = {c: probs.get(c, 0.0) for c in candidates}
    total = sum(kept.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in kept.items()}

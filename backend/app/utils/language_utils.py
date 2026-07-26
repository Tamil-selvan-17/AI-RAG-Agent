"""
Language detection utility.

Detects the language a question was written in, so the system prompt can
name it explicitly (e.g. "respond entirely in Tamil") rather than just
instructing the model to "match the question's language" and hoping it
infers correctly. Explicit naming is significantly more reliable, especially
for short or typo-heavy questions where language inference is otherwise shaky.
"""

from langdetect import DetectorFactory, LangDetectException, detect

from app.core.logging import logger

# Make detection deterministic -- langdetect is otherwise seeded randomly,
# which can flip the result of ambiguous/short text between calls.
DetectorFactory.seed = 0

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ta": "Tamil",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "pl": "Polish",
    "uk": "Ukrainian",
}


def detect_language_name(text: str) -> str | None:
    """Return a human-readable language name for the given text, or None if undetectable.

    None is returned for text too short/ambiguous to classify reliably (e.g. a single
    word, or text that's mostly numbers/punctuation) -- callers should fall back to a
    generic "match the question's language" instruction in that case rather than
    guessing wrong.
    """
    if not text or len(text.strip()) < 2:
        return None

    try:
        code = detect(text)
    except LangDetectException:
        logger.debug(f"Could not detect language for text: {text[:50]!r}")
        return None

    return _LANGUAGE_NAMES.get(code, code)

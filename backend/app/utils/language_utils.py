"""
Language detection utility.

Detects the language a question was written in, so the system prompt can
name it explicitly (e.g. "respond entirely in Tamil") rather than just
instructing the model to "match the question's language" and hoping it
infers correctly. Explicit naming is significantly more reliable, especially
for short or typo-heavy questions where language inference is otherwise shaky.
"""

import re

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
    "no": "Norwegian",
    "af": "Afrikaans",
    "et": "Estonian",
}

# Very common English function words. If a Latin-script question contains
# several of these, we trust that it's English directly rather than asking
# langdetect -- proper nouns (names, product names, resume terms) routinely
# fool langdetect's statistical model into confidently misreporting short
# English text as Norwegian, Afrikaans, Estonian, etc. (observed: "what skills
# does tamilselvan have" -> Afrikaans at 99.99% reported confidence). This
# heuristic only applies to Latin-script text; non-Latin scripts (Tamil,
# Chinese, Arabic, etc.) are already unambiguous by character set alone and
# skip straight to langdetect below.
_ENGLISH_STOPWORDS = {
    "the", "is", "are", "was", "were", "does", "do", "did", "have", "has", "had",
    "what", "how", "many", "much", "why", "who", "where", "when", "which",
    "tell", "me", "about", "in", "on", "at", "a", "an", "of", "to", "for",
    "and", "or", "but", "please", "can", "could", "you", "explain", "describe",
    "summarize", "give", "list", "show", "with", "years", "year", "experience",
    "his", "her", "he", "she", "they", "them", "it", "this", "that",
}

_LATIN_SCRIPT_RE = re.compile(r"^[a-zA-Z0-9\s.,!?'\"()\-:;]+$")


def _looks_like_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return False
    stopword_count = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    # Two or more common English function words is enough signal on typical
    # question-length text; short/ambiguous text without any is left to
    # langdetect (or the safe None fallback) rather than guessed here.
    return stopword_count >= 2


def detect_language_name(text: str) -> str | None:
    """Return a human-readable language name for the given text, or None if undetectable.

    None is returned for text too short/ambiguous to classify reliably (e.g. a single
    word, or text that's mostly numbers/punctuation) -- callers should fall back to a
    generic "match the question's language" instruction in that case rather than
    guessing wrong.
    """
    if not text or len(text.strip()) < 2:
        return None

    # Latin-script text gets an English heuristic check first (see _looks_like_english
    # docstring above for why: langdetect is unreliable here specifically, even at
    # very high reported confidence). Non-Latin-script text (different alphabet/script
    # entirely) skips this -- there's no English-vs-European-language ambiguity to
    # resolve when the script itself already rules English out.
    if _LATIN_SCRIPT_RE.match(text) and _looks_like_english(text):
        return "English"

    try:
        code = detect(text)
    except LangDetectException:
        logger.debug(f"Could not detect language for text: {text[:50]!r}")
        return None

    return _LANGUAGE_NAMES.get(code, code)

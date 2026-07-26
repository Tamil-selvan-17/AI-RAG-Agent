"""
Small-talk detection.

Generic greetings and pleasantries ("hi", "thanks", "bye") have no real
semantic content to search for -- running them through embedding + Qdrant
similarity search reliably returns *something*, since cosine similarity
between an unrelated query and any stored chunk is rarely near zero. That
"something" then gets presented to the user as a cited source, which is
misleading (e.g. "hi" citing a resume at "49% match" implies a connection
that doesn't exist).

This module catches that class of input before it ever reaches retrieval,
so the assistant can just respond naturally with no fabricated citations.
"""

import re

_GREETING_WORDS = {
    "hi", "hii", "hiya", "hello", "helo", "hey", "heyy", "hey there",
    "yo", "sup", "whats up", "what's up", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
    "morning", "evening",
    "namaste", "vanakkam", "hola",
}

_CLOSING_WORDS = {
    "bye", "goodbye", "see you", "see ya", "later", "cya",
    "thanks", "thank you", "thanks a lot", "thank you so much",
    "ty", "thx", "cool", "ok", "okay", "nice", "great", "got it",
    "cool thanks", "alright", "sounds good",
}

_ALL_SMALLTALK = _GREETING_WORDS | _CLOSING_WORDS

_CANNED_REPLIES = {
    "greeting": (
        "Hi! I'm ready to help you look through your uploaded documents. "
        "Ask me anything about them and I'll answer using what's actually in your files."
    ),
    "closing": "You're welcome! Let me know if you have any other questions about your documents.",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace, and collapse repeated letters
    (e.g. 'hiii' -> 'hi', 'heyyy' -> 'hey') so minor variations still match."""
    text = text.strip().lower()
    text = re.sub(r"[!?.,]+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "hiii" -> "hii", "heyyy" -> "heyy"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_smalltalk_reply(question: str) -> str | None:
    """Return a canned reply if the question is pure small talk, else None.

    Only matches when the *entire* message is small talk (e.g. "hi", "thanks!") --
    a message like "hi, does he know AWS?" is a real question and correctly
    falls through to normal retrieval, since it doesn't exactly match a phrase here.
    """
    normalized = _normalize(question)

    if normalized in _GREETING_WORDS:
        return _CANNED_REPLIES["greeting"]
    if normalized in _CLOSING_WORDS:
        return _CANNED_REPLIES["closing"]
    return None

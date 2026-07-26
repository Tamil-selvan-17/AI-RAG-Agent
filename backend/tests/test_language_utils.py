"""Unit tests for language detection used to keep RAG answers in a consistent language."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.language_utils import detect_language_name  # noqa: E402


def test_detects_english():
    assert detect_language_name("how many years of experience does he have") == "English"


def test_detects_english_with_typos():
    assert detect_language_name("how many year of exprence he have") == "English"


def test_detects_tamil():
    assert detect_language_name("எத்தனை ஆண்டு அனுபவம் உள்ளது") == "Tamil"


def test_detects_chinese():
    assert detect_language_name("他已经拥有超过3年的专业工作经验") in (
        "Chinese (Simplified)",
        "Chinese (Traditional)",
    )


def test_returns_none_for_empty_text():
    assert detect_language_name("") is None
    assert detect_language_name("   ") is None


def test_returns_none_for_too_short_text():
    assert detect_language_name("a") is None

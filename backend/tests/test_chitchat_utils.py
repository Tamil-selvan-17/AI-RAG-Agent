"""Unit tests for small-talk detection, ensuring greetings never get treated as document questions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.chitchat_utils import detect_smalltalk_reply  # noqa: E402


def test_detects_plain_greeting():
    assert detect_smalltalk_reply("hi") is not None
    assert detect_smalltalk_reply("Hi") is not None
    assert detect_smalltalk_reply("hello!") is not None


def test_detects_repeated_letter_variants():
    assert detect_smalltalk_reply("hii") is not None
    assert detect_smalltalk_reply("heyyy") is not None


def test_detects_closing_phrases():
    assert detect_smalltalk_reply("thanks") is not None
    assert detect_smalltalk_reply("thank you!") is not None
    assert detect_smalltalk_reply("bye") is not None
    assert detect_smalltalk_reply("ok") is not None


def test_does_not_flag_real_questions():
    assert detect_smalltalk_reply("how many years of experience does he have") is None
    assert detect_smalltalk_reply("hi, does he know AWS?") is None
    assert detect_smalltalk_reply("okay so what skills does he have") is None
    assert detect_smalltalk_reply("what is in the document") is None

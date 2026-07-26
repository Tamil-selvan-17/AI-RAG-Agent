"""Unit tests for parsing Gemini's 429 error responses into specific, actionable messages."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.gemini_service import _describe_rate_limit  # noqa: E402


def test_per_minute_limit_includes_retry_delay():
    body = json.dumps({
        "error": {
            "message": "Resource has been exhausted (e.g. check quota).",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}
            ],
        }
    })
    message = _describe_rate_limit(body)
    assert "23s" in message
    assert "daily" not in message.lower()


def test_per_day_limit_mentions_midnight_reset():
    body = json.dumps({
        "error": {"message": "Quota exceeded for quota metric GenerateContentPerDay"}
    })
    message = _describe_rate_limit(body)
    assert "midnight" in message.lower()
    assert "aistudio.google.com/usage" in message


def test_unparseable_body_falls_back_gracefully():
    message = _describe_rate_limit("not json at all")
    assert "rate limit" in message.lower()
    assert message  # never empty

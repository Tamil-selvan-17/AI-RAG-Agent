"""Tests that transient provider failures (per-minute 429s) are actually retried."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.gemini_service import GeminiService  # noqa: E402

_PER_MINUTE_429 = json.dumps({
    "error": {
        "message": "Resource has been exhausted.",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0.01s"}],
    }
})
_PER_DAY_429 = json.dumps({"error": {"message": "Quota exceeded for metric GenerateContentPerDay"}})


def _service_with_responses(monkeypatch, responses: list[httpx.Response]) -> tuple[GeminiService, list]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses[min(len(requests) - 1, len(responses) - 1)]

    service = GeminiService()
    service.api_key = "test-key"
    monkeypatch.setattr(
        service,
        "_client",
        lambda: httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
            headers={"x-goog-api-key": service.api_key},
        ),
    )
    return service, requests


@pytest.mark.asyncio
async def test_per_minute_rate_limit_is_retried_then_succeeds(monkeypatch):
    service, requests = _service_with_responses(
        monkeypatch,
        [
            httpx.Response(429, text=_PER_MINUTE_429),
            httpx.Response(200, json={"embedding": {"values": [0.1, 0.2]}}),
        ],
    )
    assert await service.generate_embedding("hello") == [0.1, 0.2]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_daily_rate_limit_is_not_retried(monkeypatch):
    service, requests = _service_with_responses(monkeypatch, [httpx.Response(429, text=_PER_DAY_429)])
    with pytest.raises(HTTPException) as exc_info:
        await service.generate_embedding("hello")
    assert exc_info.value.status_code == 429
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_api_key_is_sent_in_header_not_url(monkeypatch):
    service, requests = _service_with_responses(
        monkeypatch, [httpx.Response(200, json={"embedding": {"values": [0.1]}})]
    )
    await service.generate_embedding("hello")
    assert "key=" not in str(requests[0].url)
    assert requests[0].headers["x-goog-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_batch_embedding_uses_document_task_type_in_one_request(monkeypatch):
    service, requests = _service_with_responses(
        monkeypatch,
        [httpx.Response(200, json={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]})],
    )
    vectors = await service.generate_embeddings_batch(["a", "b"])
    assert vectors == [[0.1], [0.2]]
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert {r["taskType"] for r in body["requests"]} == {"RETRIEVAL_DOCUMENT"}

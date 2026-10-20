"""Tests for request IDs and optional OS certificate store support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import tls  # noqa: E402
from app.core.config import get_settings  # noqa: E402


def test_every_response_has_a_request_id(anon_client):
    response = anon_client.get("/health/live")
    assert len(response.headers["x-request-id"]) == 12


def test_incoming_request_id_is_reused(anon_client):
    response = anon_client.get("/health/live", headers={"X-Request-ID": "trace-abc.123"})
    assert response.headers["x-request-id"] == "trace-abc.123"


def test_unsafe_incoming_request_id_is_replaced(anon_client):
    response = anon_client.get("/health/live", headers={"X-Request-ID": "bad id\twith spaces"})
    assert response.headers["x-request-id"] != "bad id\twith spaces"
    assert len(response.headers["x-request-id"]) == 12


def test_system_certs_are_injected_only_when_enabled(monkeypatch):
    import truststore

    calls = []
    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))
    monkeypatch.setattr(tls, "_configured", False)

    monkeypatch.setattr(get_settings(), "use_system_certs", False)
    tls.configure_tls()
    assert calls == []

    monkeypatch.setattr(get_settings(), "use_system_certs", True)
    tls.configure_tls()
    tls.configure_tls()  # idempotent
    assert calls == [1]

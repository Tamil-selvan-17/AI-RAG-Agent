"""Tests for single-user login and route protection."""

import time

import pytest

from app.core import auth
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _reset_failed_attempts():
    auth._failed_attempts.clear()
    yield
    auth._failed_attempts.clear()


def _login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def test_login_with_correct_credentials_returns_token(anon_client):
    settings = get_settings()
    response = _login(anon_client, settings.admin_username, settings.admin_password)
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == settings.admin_username
    assert body["access_token"]
    assert body["expires_at"] > time.time()


def test_login_with_wrong_password_is_rejected(anon_client):
    response = _login(anon_client, get_settings().admin_username, "wrong-password")
    assert response.status_code == 401


def test_protected_routes_require_login(anon_client):
    assert anon_client.get("/api/documents").status_code == 401
    assert anon_client.delete("/api/documents").status_code == 401
    assert anon_client.post("/api/chat", json={"question": "hi"}).status_code == 401
    assert anon_client.post("/api/chat/stream", json={"question": "hi"}).status_code == 401


def test_health_stays_public(anon_client):
    assert anon_client.get("/health").status_code == 200


def test_me_returns_logged_in_user(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["username"] == get_settings().admin_username


def test_tampered_token_is_rejected(anon_client):
    token, _ = auth.create_token(get_settings().admin_username)
    encoded, signature = token.split(".", 1)
    forged = f"{encoded}.{'A' * len(signature)}"
    response = anon_client.get("/api/documents", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_expired_token_is_rejected(monkeypatch):
    token, _ = auth.create_token(get_settings().admin_username)
    monkeypatch.setattr(auth.time, "time", lambda: time.time_ns() / 1e9 + 10 * 24 * 3600)
    assert auth.verify_token(token) is None


def test_repeated_failures_lock_out_login(anon_client):
    settings = get_settings()
    for _ in range(5):
        assert _login(anon_client, settings.admin_username, "bad").status_code == 401
    # Even the correct password is refused during the lockout window.
    response = _login(anon_client, settings.admin_username, settings.admin_password)
    assert response.status_code == 429

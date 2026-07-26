"""Tests for POST /api/chat and GET /api/chat/history/{conversation_id}."""


def test_chat_returns_answer(client):
    response = client.post("/api/chat", json={"question": "What is in the document?"})
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv-abc"
    assert body["answer"] == "This is the answer."
    assert body["sources"] == []


def test_chat_rejects_empty_question(client):
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_stream_returns_sse_events(client):
    response = client.post("/api/chat/stream", json={"question": "What is in the document?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "data: " in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "conv-stream" in body


def test_chat_history_not_found_returns_404(client):
    response = client.get("/api/chat/history/nonexistent-conversation")
    assert response.status_code == 404

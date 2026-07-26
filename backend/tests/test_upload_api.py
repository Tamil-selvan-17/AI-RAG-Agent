"""Tests for POST /api/documents/upload and GET /api/documents."""

import io


def test_upload_valid_txt_file(client):
    file_content = b"This is a sample document used for testing the upload pipeline."
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "sample.txt"
    assert body["status"] == "ready"
    assert body["chunk_count"] == 3


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 400


def test_list_documents(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["filename"] == "sample.txt"


def test_delete_single_document(client):
    response = client.delete("/api/documents/doc-123")
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-123"


def test_delete_nonexistent_document_returns_404(client):
    response = client.delete("/api/documents/does-not-exist")
    assert response.status_code == 404


def test_delete_all_documents(client):
    response = client.delete("/api/documents")
    assert response.status_code == 200
    assert response.json()["count"] == 1

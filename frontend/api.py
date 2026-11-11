"""
Frontend API communication client.
"""

from typing import Any
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"


def upload_files(files_payload: list) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}/upload", files=files_payload, timeout=120)
    response.raise_for_status()
    return response.json()


def fetch_documents() -> list[str]:
    response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
    response.raise_for_status()
    return response.json().get("documents", [])


def delete_document(document_name: str) -> dict[str, Any]:
    response = requests.delete(f"{API_BASE_URL}/documents/{document_name}", timeout=10)
    response.raise_for_status()
    return response.json()


def send_chat_message(
    question: str,
    history: list[dict] | None = None,
    document_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "history": history,
        "document_id": document_id,
        "top_k": top_k,
    }
    response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


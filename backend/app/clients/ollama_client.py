import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


def chat_with_ollama(
    system_prompt: str,
    user_prompt: str,
) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaConnectionError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}"
        ) from exc

    data = response.json()

    message = data.get("message", {})
    content = message.get("content")

    if not content:
        raise OllamaConnectionError(
            "Ollama returned an empty response."
        )

    return content

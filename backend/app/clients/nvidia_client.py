import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

import httpx

from app.clients.exceptions import LLMClientError
from app.core.config import settings
from app.prompts.assistance_prompt import build_assistance_prompt
from app.services.image_service import ProcessedImage

logger = logging.getLogger(__name__)


class NvidiaClientError(LLMClientError):
    pass


def _mask_api_key(text: str) -> str:
    """Mask the API key for safe logging."""
    if not text:
        return ""
    if len(text) <= 8:
        return text[:2] + "***"
    return text[:4] + "***" + text[-4:]


def _get_api_key() -> str:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise NvidiaClientError(
            "NVIDIA_API_KEY is not set. Set it as a system environment variable "
            "when using LLM_PROVIDER=nvidia."
        )
    return api_key


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from the model response text."""
    cleaned_text = text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    cleaned_text = cleaned_text.strip()

    try:
        return json.loads(cleaned_text)

    except json.JSONDecodeError:
        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            logger.info(
                "Model output is not JSON — wrapping as narrative summary "
                "(length=%d chars)",
                len(cleaned_text),
            )
            return {
                "intent": "",
                "product": "",
                "summary": cleaned_text,
                "possible_causes": [],
                "steps": [],
                "warning": None,
                "escalation_required": False,
            }

        possible_json = cleaned_text[start: end + 1]

        try:
            return json.loads(possible_json)
        except json.JSONDecodeError as exc:
            logger.info(
                "Model output contains JSON-like delimiters but is invalid — "
                "wrapping as narrative summary (length=%d chars)",
                len(cleaned_text),
            )
            return {
                "intent": "",
                "product": "",
                "summary": cleaned_text,
                "possible_causes": [],
                "steps": [],
                "warning": None,
                "escalation_required": False,
            }


async def generate_assistance(
    question: str,
    product: str,
    assistance_type: str,
    context: str | None = None,
) -> dict[str, Any]:
    if context:
        context_len = len(context)
        logger.info(
            "Including RAG context — length=%d chars",
            context_len,
        )
    else:
        logger.info("No RAG context provided")

    prompt = build_assistance_prompt(
        question=question,
        product=product,
        assistance_type=assistance_type,
        context=context,
    )

    api_key = _get_api_key()
    url = settings.nvidia_chat_url
    model = settings.nvidia_model
    timeout = 60.0

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a product assistance assistant. "
                    "Return only valid JSON. Do not include Markdown code fences. "
                    "Do not include any text before or after the JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "stream": False,
    }

    logger.info(
        "NVIDIA request started — model=%s assistance_type=%s context_provided=%s",
        model,
        assistance_type,
        context is not None,
    )

    start_time = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            elapsed = time.monotonic() - start_time
            logger.info(
                "NVIDIA response — status=%s duration=%.1fs",
                response.status_code,
                elapsed,
            )

            if response.status_code == 401:
                raise NvidiaClientError(
                    "NVIDIA API authentication failed. Check that NVIDIA_API_KEY is valid."
                )

            if response.status_code == 403:
                raise NvidiaClientError(
                    "NVIDIA API access denied (403). The API key may not have "
                    "access to the requested model."
                )

            if response.status_code == 404:
                raise NvidiaClientError(
                    f"NVIDIA model '{model}' not found. Check NVIDIA_MODEL."
                )

            if response.status_code == 429:
                raise NvidiaClientError(
                    "NVIDIA API rate limit exceeded. Try again later."
                )

            response.raise_for_status()

    except httpx.ConnectError as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA connection failed — duration=%.1fs url=%s",
            elapsed,
            url,
        )
        raise NvidiaClientError(
            "Cannot connect to the NVIDIA API. Check NVIDIA_BASE_URL and your network connection."
        ) from exc

    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA request timed out — duration=%.1fs timeout=%ss",
            elapsed,
            timeout,
        )
        raise NvidiaClientError(
            f"The NVIDIA API did not respond within {timeout} seconds."
        ) from exc

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - start_time
        status = exc.response.status_code
        body = exc.response.text[:500]
        logger.error(
            "NVIDIA HTTP error — status=%s duration=%.1fs body=%s",
            status,
            elapsed,
            body,
        )
        raise NvidiaClientError(
            f"NVIDIA API returned HTTP {status}. Check the configuration and try again."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected NVIDIA error")
        raise NvidiaClientError(f"Unexpected error: {exc}") from exc

    try:
        response_data = response.json()
    except Exception as exc:
        logger.exception("Failed to parse NVIDIA response JSON")
        raise NvidiaClientError("Failed to parse NVIDIA response as JSON.") from exc
    choices = response_data.get("choices")

    if not choices or not isinstance(choices, list) or len(choices) == 0:
        logger.error(
            "NVIDIA response missing choices — raw=%s",
            str(response_data)[:300],
        )
        raise NvidiaClientError(
            "The NVIDIA API returned an unexpected response format (missing choices)."
        )

    message = choices[0].get("message", {})
    generated_text = message.get("content")

    if not generated_text:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA returned empty content — duration=%.1fs raw=%s",
            elapsed,
            str(response_data)[:300],
        )
        raise NvidiaClientError(
            "The NVIDIA model returned an empty response."
        )

    logger.info(
        "NVIDIA response parsed — length=%d chars duration=%.1fs",
        len(generated_text),
        time.monotonic() - start_time,
    )

    return extract_json(generated_text)


async def generate_assistance_with_image(
    question: str,
    product: str,
    assistance_type: str,
    context: str | None = None,
    image: ProcessedImage | None = None,
) -> dict[str, Any]:
    import base64

    if image is None:
        raise NvidiaClientError("generate_assistance_with_image called without an image.")

    vision_model = settings.nvidia_vision_model
    if not vision_model:
        raise NvidiaClientError(
            "Image provided but NVIDIA_VISION_MODEL is not configured. "
            "Set NVIDIA_VISION_MODEL in your .env file."
        )

    prompt = build_assistance_prompt(
        question=question,
        product=product,
        assistance_type=assistance_type,
        context=context,
    )

    from app.prompts.assistance_prompt import inject_image_instruction
    prompt = inject_image_instruction(prompt)

    b64_data = base64.b64encode(image.data).decode("ascii")
    data_uri = f"data:{image.media_type};base64,{b64_data}"

    api_key = _get_api_key()
    url = settings.nvidia_chat_url
    timeout = 60.0

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a product assistance assistant. "
                    "Return only valid JSON. Do not include Markdown code fences. "
                    "Do not include any text before or after the JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
        "stream": False,
    }

    logger.info(
        "NVIDIA vision request started — model=%s assistance_type=%s image=%s",
        vision_model,
        assistance_type,
        f"{image.width}x{image.height}",
    )

    start_time = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            elapsed = time.monotonic() - start_time
            logger.info(
                "NVIDIA vision response — status=%s duration=%.1fs",
                response.status_code,
                elapsed,
            )

            if response.status_code == 401:
                raise NvidiaClientError(
                    "NVIDIA API authentication failed. Check that NVIDIA_API_KEY is valid."
                )

            if response.status_code == 403:
                raise NvidiaClientError(
                    "NVIDIA API access denied (403). The API key may not have "
                    "access to the requested model."
                )

            if response.status_code == 404:
                raise NvidiaClientError(
                    f"NVIDIA vision model '{vision_model}' not found. Check NVIDIA_VISION_MODEL."
                )

            if response.status_code == 429:
                raise NvidiaClientError(
                    "NVIDIA API rate limit exceeded. Try again later."
                )

            response.raise_for_status()

    except httpx.ConnectError as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA vision connection failed — duration=%.1fs url=%s",
            elapsed,
            url,
        )
        raise NvidiaClientError(
            "Cannot connect to the NVIDIA API. Check NVIDIA_BASE_URL and your network connection."
        ) from exc

    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA vision request timed out — duration=%.1fs timeout=%ss",
            elapsed,
            timeout,
        )
        raise NvidiaClientError(
            f"The NVIDIA API did not respond within {timeout} seconds."
        ) from exc

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - start_time
        status = exc.response.status_code
        body = exc.response.text[:500]
        logger.error(
            "NVIDIA vision HTTP error — status=%s duration=%.1fs body=%s",
            status,
            elapsed,
            body,
        )
        raise NvidiaClientError(
            f"NVIDIA API returned HTTP {status}. Check the configuration and try again."
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected NVIDIA vision error")
        raise NvidiaClientError(f"Unexpected error: {exc}") from exc

    try:
        response_data = response.json()
    except Exception as exc:
        logger.exception("Failed to parse NVIDIA vision response JSON")
        raise NvidiaClientError("Failed to parse NVIDIA vision response as JSON.") from exc

    choices = response_data.get("choices")
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        logger.error(
            "NVIDIA vision response missing choices — raw=%s",
            str(response_data)[:300],
        )
        raise NvidiaClientError(
            "The NVIDIA API returned an unexpected response format (missing choices)."
        )

    message = choices[0].get("message", {})
    generated_text = message.get("content")
    if not generated_text:
        elapsed = time.monotonic() - start_time
        logger.error(
            "NVIDIA vision returned empty content — duration=%.1fs raw=%s",
            elapsed,
            str(response_data)[:300],
        )
        raise NvidiaClientError("The NVIDIA vision model returned an empty response.")

    logger.info(
        "NVIDIA vision response parsed — length=%d chars duration=%.1fs",
        len(generated_text),
        time.monotonic() - start_time,
    )

    return extract_json(generated_text)

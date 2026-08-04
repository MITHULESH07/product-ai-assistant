import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.clients.exceptions import LLMClientError
from app.core.config import settings
from app.prompts.assistance_prompt import build_assistance_prompt
from app.services.image_service import ProcessedImage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({400, 408, 429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (1.0, 2.0)
_NUM_CTX = 8192


async def _post_with_retry(url: str, payload: dict, timeout: float) -> httpx.Response:
    """POST to Ollama, retrying transient status codes with backoff."""
    for attempt in range(_MAX_ATTEMPTS):
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)

        status = response.status_code
        if status in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
            logger.warning(
                "Ollama transient response — status=%s attempt=%d/%d retrying in %.0fs",
                status,
                attempt + 1,
                _MAX_ATTEMPTS,
                _RETRY_BACKOFF_SECONDS[attempt],
            )
            await asyncio.sleep(_RETRY_BACKOFF_SECONDS[attempt])
            continue

        return response


class OllamaClientError(LLMClientError):
    pass


def extract_json(text: str) -> dict[str, Any]:
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

    url = settings.ollama_generate_url
    model = settings.ollama_model
    timeout = settings.ollama_timeout

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": _NUM_CTX,
        },
    }

    logger.info(
        "Ollama request started — model=%s assistance_type=%s context_provided=%s",
        model,
        assistance_type,
        context is not None,
    )

    start_time = time.monotonic()

    try:
        response = await _post_with_retry(url, payload, timeout)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Ollama response — status=%s duration=%.1fs",
            response.status_code,
            elapsed,
        )
        logger.debug("Ollama raw response (first 500): %s", response.text[:500])

        response.raise_for_status()

    except httpx.ConnectError as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama connection failed — duration=%.1fs url_masked=%s",
            elapsed,
            _mask_url(url),
        )
        raise OllamaClientError(
            "Cannot connect to the remote Ollama server. "
            "Ensure the teammate's laptop is powered on, "
            "Ollama is running, and both laptops are on the same network."
        ) from exc

    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama request timed out — duration=%.1fs timeout=%ss",
            elapsed,
            timeout,
        )
        raise OllamaClientError(
            f"The Ollama model did not respond within {timeout} seconds. "
            "Try a smaller model, reduce prompt length, "
            "or increase OLLAMA_TIMEOUT_SECONDS."
        ) from exc

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - start_time
        status = exc.response.status_code
        body = exc.response.text[:500]
        logger.error(
            "Ollama HTTP error — status=%s duration=%.1fs",
            status,
            elapsed,
        )

        if status == 404:
            try:
                error_data = json.loads(body)
                ollama_msg = error_data.get("error", body)
            except (json.JSONDecodeError, TypeError):
                ollama_msg = body

            raise OllamaClientError(
                f"The configured Ollama model '{model}' is not available "
                f"on the remote server. Error: {ollama_msg}"
            ) from exc

        raise OllamaClientError(
            f"Ollama returned an unexpected HTTP {status} response."
        ) from exc

    response_data = response.json()
    generated_text = response_data.get("response")

    if not generated_text:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama returned empty response — duration=%.1fs raw=%s",
            elapsed,
            str(response_data)[:300],
        )
        raise OllamaClientError(
            "The Ollama model returned an empty response. "
            "This may indicate the model failed to generate output."
        )

    logger.info(
        "Ollama response parsed — length=%d chars duration=%.1fs",
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
        raise OllamaClientError("generate_assistance_with_image called without an image.")

    vision_model = settings.ollama_vision_model
    if not vision_model:
        raise OllamaClientError(
            "Image provided but OLLAMA_VISION_MODEL is not configured. "
            "Set OLLAMA_VISION_MODEL in your .env file."
        )

    # The vision model only describes the image; the full RAG context is
    # injected later in the structuring step (vision models have small
    # context windows and prompt + image tokens must stay under the limit).
    prompt = build_assistance_prompt(
        question=question,
        product=product,
        assistance_type=assistance_type,
        context=None,
    )

    from app.prompts.assistance_prompt import inject_image_instruction
    prompt = inject_image_instruction(prompt)

    b64_data = base64.b64encode(image.data).decode("ascii")

    url = settings.ollama_generate_url
    timeout = settings.ollama_timeout

    payload = {
        "model": vision_model,
        "prompt": prompt,
        "images": [b64_data],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": _NUM_CTX,
        },
    }

    logger.info(
        "Ollama vision request started — model=%s assistance_type=%s image=%s",
        vision_model,
        assistance_type,
        f"{image.width}x{image.height}",
    )

    start_time = time.monotonic()

    try:
        response = await _post_with_retry(url, payload, timeout)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Ollama vision response — status=%s duration=%.1fs",
            response.status_code,
            elapsed,
        )
        logger.debug("Ollama vision raw response (first 500): %s", response.text[:500])

        response.raise_for_status()

    except httpx.ConnectError as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama vision connection failed — duration=%.1fs url_masked=%s",
            elapsed,
            _mask_url(url),
        )
        raise OllamaClientError(
            "Cannot connect to the remote Ollama server. "
            "Ensure the teammate's laptop is powered on, "
            "Ollama is running, and both laptops are on the same network."
        ) from exc

    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama vision request timed out — duration=%.1fs timeout=%ss",
            elapsed,
            timeout,
        )
        raise OllamaClientError(
            f"The Ollama model did not respond within {timeout} seconds. "
            "Try a smaller model, reduce prompt length, "
            "or increase OLLAMA_TIMEOUT_SECONDS."
        ) from exc

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - start_time
        status = exc.response.status_code
        body = exc.response.text[:500]
        logger.error(
            "Ollama vision HTTP error — status=%s duration=%.1fs",
            status,
            elapsed,
        )

        if status == 404:
            try:
                error_data = json.loads(body)
                ollama_msg = error_data.get("error", body)
            except (json.JSONDecodeError, TypeError):
                ollama_msg = body

            raise OllamaClientError(
                f"The configured vision model '{vision_model}' is not available "
                f"on the remote server. Error: {ollama_msg}"
            ) from exc

        raise OllamaClientError(
            f"Ollama returned an unexpected HTTP {status} response."
        ) from exc

    response_data = response.json()
    generated_text = response_data.get("response")

    if not generated_text:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Ollama vision returned empty response — duration=%.1fs raw=%s",
            elapsed,
            str(response_data)[:300],
        )
        raise OllamaClientError(
            "The Ollama vision model returned an empty response. "
            "This may indicate the model failed to generate output."
        )

    logger.info(
        "Ollama vision response parsed — length=%d chars duration=%.1fs",
        len(generated_text),
        time.monotonic() - start_time,
    )

    parsed = extract_json(generated_text)

    if (
        not parsed.get("intent")
        and not parsed.get("possible_causes")
        and not parsed.get("steps")
        and parsed.get("summary")
    ):
        logger.info("Vision model returned a narrative — structuring via text model")
        structured = await _structure_narrative(
            narrative=str(parsed["summary"]),
            question=question,
            product=product,
            assistance_type=assistance_type,
            context=context,
        )
        if structured is not None:
            return structured

    return parsed


async def _structure_narrative(
    narrative: str,
    question: str,
    product: str,
    assistance_type: str,
    context: str | None = None,
) -> dict[str, Any] | None:
    """Convert a free-form vision-model narrative into a structured response.

    Small vision models (e.g. moondream) cannot emit reliable JSON and have a
    2048-token context window, so a fast text model reformats the narrative —
    together with the full RAG context — into the standard schema.

    Args:
        narrative: The vision model's free-form image description.
        question: The user's original question.
        product: The selected product.
        assistance_type: The assistance-type selector value.
        context: The full retrieved RAG context, if available.

    Returns:
        A structured dict, or ``None`` if the reformatting call fails.
    """
    model = settings.ollama_model
    intent_hint = assistance_type if assistance_type != "auto" else "general"

    context_section = ""
    if context:
        context_section = (
            "\n\nRelevant documentation:\n"
            f"{context}\n\n"
            "GROUNDING RULES:\n"
            "- Use the retrieved manual context as your PRIMARY source.\n"
            "- Do NOT invent product-specific voltages, limits, procedures, "
            "warnings, part numbers, pin mappings, or specifications that are "
            "absent from the retrieved context.\n"
            "- When the context does not contain enough information, explicitly "
            "state that the indexed manual does not provide enough information.\n"
            "- Do NOT claim that a statement came from the manual unless "
            "supported by the supplied context."
        )

    prompt = (
        "You are a JSON formatter for a product-support assistant. "
        "Reformat the description below into one valid JSON object. "
        "Write real, specific, actionable content for every field — do NOT "
        "repeat the template hints or placeholder text below; they are only "
        "examples of the required format. "
        "Do not invent technical facts that are not present. "
        "Return only the JSON object — no code fences, no commentary.\n\n"
        f"Product: {product}\n"
        f"User question: {question}\n"
        f"Image description: {narrative}"
        f"{context_section}\n\n"
        "Output exactly this shape:\n"
        "{\n"
        f'  "intent": "{intent_hint}",\n'
        f'  "product": "{product}",\n'
        '  "summary": "a concise, specific summary that incorporates the image description",\n'
        '  "possible_causes": ["an actionable cause suggested by the image or question"],\n'
        '  "steps": ["a clear, numbered troubleshooting step"],\n'
        '  "warning": "a relevant safety warning, or null",\n'
        '  "escalation_required": false\n'
        "}"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": _NUM_CTX},
    }

    try:
        response = await _post_with_retry(
            settings.ollama_generate_url,
            payload,
            min(settings.ollama_timeout, 120),
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response")
    except Exception as exc:
        logger.warning(
            "Vision narrative structuring failed — falling back to raw summary: %s",
            exc,
        )
        return None

    if not text:
        logger.warning("Vision narrative structuring returned empty text")
        return None

    structured = extract_json(text)
    if not isinstance(structured, dict) or not structured.get("summary"):
        logger.warning("Vision narrative structuring returned unusable output")
        return None

    if not structured.get("intent"):
        structured["intent"] = intent_hint

    logger.info(
        "Vision narrative structured — model=%s summary_len=%d causes=%d steps=%d",
        model,
        len(str(structured.get("summary", ""))),
        len(structured.get("possible_causes") or []),
        len(structured.get("steps") or []),
    )
    return structured


async def check_ollama_health() -> dict[str, Any]:
    base_url = settings.ollama_base_url
    model = settings.ollama_model
    tags_url = f"{base_url}/api/tags"

    result: dict[str, Any] = {
        "status": "unknown",
        "server_reachable": False,
        "configured_model": model,
        "model_available": False,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(tags_url)

        if response.status_code != 200:
            result["status"] = "unhealthy"
            return result

        data = response.json()
        models = data.get("models", [])
        installed_models = [m["name"] for m in models if "name" in m]

        result["server_reachable"] = True
        result["model_available"] = model in installed_models
        result["status"] = "healthy" if result["model_available"] else "degraded"

        if not result["model_available"]:
            logger.warning(
                "Configured model '%s' not found on remote server. "
                "Available: %s",
                model,
                installed_models,
            )

    except httpx.ConnectError:
        result["status"] = "unreachable"
        logger.error("Ollama health check — connection refused to %s", _mask_url(base_url))

    except httpx.TimeoutException:
        result["status"] = "timeout"
        logger.error("Ollama health check — timed out connecting to %s", _mask_url(base_url))

    except Exception as exc:
        result["status"] = "error"
        logger.error("Ollama health check — unexpected error: %s", exc)

    return result


def _mask_url(url: str) -> str:
    for prefix in ("http://", "https://"):
        if url.startswith(prefix):
            rest = url[len(prefix):]
            if ":" in rest:
                host, port = rest.split(":", 1)
                return f"{prefix}[masked]:{port}"
            return f"{prefix}[masked]"
    return "[masked]"

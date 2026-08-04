"""Diagnostic script to test NVIDIA connection.

Usage:
    cd backend
    python scripts/test_nvidia_connection.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("test_nvidia")


def check_env() -> bool:
    logger.info("Checking environment configuration...")

    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded .env from: %s", dotenv_path)
    else:
        logger.warning("No .env file found — using system environment only")

    provider = os.getenv("LLM_PROVIDER", "ollama")
    logger.info("LLM_PROVIDER: %s", provider)

    base_url = os.getenv("NVIDIA_BASE_URL", "(not set)")
    logger.info("NVIDIA_BASE_URL: %s", base_url)

    model = os.getenv("NVIDIA_MODEL", "(not set)")
    logger.info("NVIDIA_MODEL: %s", model)

    key_present = bool(os.getenv("NVIDIA_API_KEY"))
    logger.info("NVIDIA_API_KEY present: %s", key_present)
    if key_present:
        logger.info("NVIDIA_API_KEY length: %d", len(os.getenv("NVIDIA_API_KEY", "")))
    else:
        logger.error("NVIDIA_API_KEY is NOT set in system environment!")
        return False

    from app.core.config import settings

    try:
        settings.validate()
    except RuntimeError as e:
        logger.error("Settings validation failed: %s", e)
        return False

    logger.info("Configured provider: %s", settings.llm_provider)
    logger.info("NVIDIA chat URL: %s", settings.nvidia_chat_url)
    logger.info("NVIDIA model: %s", settings.nvidia_model)
    logger.info("Ollama base URL: %s", settings.ollama_base_url)
    logger.info("Ollama model: %s", settings.ollama_model)
    logger.info("Ollama embedding model: %s", settings.ollama_embedding_model)

    if settings.llm_provider != "nvidia":
        logger.warning(
            "Provider is '%s', not 'nvidia'. Set LLM_PROVIDER=nvidia in .env.",
            settings.llm_provider,
        )

    expected_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    if settings.nvidia_chat_url != expected_url:
        logger.warning(
            "Unexpected URL: %s (expected: %s)",
            settings.nvidia_chat_url,
            expected_url,
        )

    return True


async def check_connection() -> bool:
    logger.info("Sending test request to NVIDIA API...")

    api_key = os.getenv("NVIDIA_API_KEY", "")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    model = "meta/llama-3.1-8b-instruct"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with just OK and nothing else."}
        ],
        "temperature": 0.2,
        "max_tokens": 50,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            logger.info("HTTP status: %s", response.status_code)

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    logger.info("SUCCESS — model replied: %s", content.strip())
                    return True
                else:
                    logger.error("No choices in response: %s", str(data)[:300])
                    return False
            elif response.status_code == 401:
                logger.error("AUTH FAILED — invalid NVIDIA_API_KEY")
            elif response.status_code == 403:
                logger.error("ACCESS DENIED — key lacks model access")
            elif response.status_code == 404:
                logger.error("NOT FOUND — check NVIDIA_BASE_URL and NVIDIA_MODEL")
                logger.error("Response body: %s", response.text[:500])
            elif response.status_code == 429:
                logger.error("RATE LIMITED — try again later")
            else:
                logger.error("HTTP %s — %s", response.status_code, response.text[:500])
            return False

    except httpx.ConnectError as e:
        logger.error("CONNECTION FAILED — %s", e)
        return False
    except httpx.TimeoutException:
        logger.error("TIMEOUT — request exceeded 30 seconds")
        return False
    except Exception as e:
        logger.exception("UNEXPECTED ERROR — %s", e)
        return False


async def check_via_factory() -> bool:
    logger.info("Testing via provider factory...")

    from app.clients.llm_factory import generate_assistance

    try:
        result = await generate_assistance(
            question="Reply with OK if you receive this.",
            product="Diagnostic Check",
            assistance_type="troubleshooting",
        )
        logger.info("Factory result summary: %s", result.get("summary", "")[:100])
        logger.info("Factory returned all expected fields: %s", all(k in result for k in ("summary", "possible_causes", "steps")))
        return True
    except Exception as e:
        logger.exception("Factory test failed: %s", e)
        return False


async def main():
    logger.info("=" * 60)
    logger.info("NVIDIA Connection Diagnostic")
    logger.info("=" * 60)

    env_ok = check_env()
    if not env_ok:
        logger.error("Environment check FAILED — aborting")
        return False

    logger.info("")
    logger.info("--- Step 1: Direct connection test ---")
    direct_ok = await check_connection()
    if not direct_ok:
        logger.error("Direct connection FAILED")

    logger.info("")
    logger.info("--- Step 2: Factory integration test ---")
    factory_ok = await check_via_factory()
    if not factory_ok:
        logger.error("Factory integration FAILED")

    logger.info("")
    logger.info("=" * 60)
    if direct_ok and factory_ok:
        logger.info("ALL CHECKS PASSED — NVIDIA integration is working.")
        return True
    else:
        logger.warning("SOME CHECKS FAILED — review the logs above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

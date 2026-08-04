import logging
from typing import Any

from app.clients.exceptions import LLMClientError
from app.services.image_service import ProcessedImage

logger = logging.getLogger(__name__)

_generate_fn = None
_vision_generate_fn = None


def _resolve_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in ("ollama", "nvidia"):
        return normalized
    raise ValueError(
        f"Unsupported LLM provider '{provider}'. "
        f"Supported values: 'ollama', 'nvidia'."
    )


async def generate_assistance(
    question: str,
    product: str,
    assistance_type: str,
    context: str | None = None,
    image: ProcessedImage | None = None,
) -> dict[str, Any]:
    global _generate_fn, _vision_generate_fn

    from app.core.config import settings

    provider = _resolve_provider(settings.llm_provider)

    if image is not None:
        if _vision_generate_fn is None:
            if provider == "ollama":
                from app.clients.ollama_client import generate_assistance_with_image as fn
            else:
                from app.clients.nvidia_client import generate_assistance_with_image as fn
            _vision_generate_fn = fn

            logger.info(
                "LLM provider selected for image analysis: %s",
                provider,
            )

        return await _vision_generate_fn(
            question=question,
            product=product,
            assistance_type=assistance_type,
            context=context,
            image=image,
        )

    if _generate_fn is None:
        if provider == "ollama":
            from app.clients.ollama_client import generate_assistance as fn
        else:
            from app.clients.nvidia_client import generate_assistance as fn

        _generate_fn = fn

        logger.info("LLM provider selected: %s", provider)

        if provider == "nvidia":
            from app.core.config import settings as _s

            logger.info(
                "NVIDIA — url=%s model=%s",
                _s.nvidia_chat_url,
                _s.nvidia_model,
            )

    return await _generate_fn(
        question=question,
        product=product,
        assistance_type=assistance_type,
        context=context,
    )

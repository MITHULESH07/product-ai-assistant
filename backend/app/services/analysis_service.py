import logging
import re

from app.agents.agent_router import route_to_agent
from app.rag.embedding_client import OllamaEmbeddingClient
from app.rag.retrieval_service import RetrievalService, RetrievedChunk
from app.rag.vector_store import ChromaVectorStore
from app.services.image_service import ProcessedImage

logger = logging.getLogger(__name__)

_retrieval_service: RetrievalService | None = None
_vector_store: ChromaVectorStore | None = None


def _get_retrieval_components() -> tuple[RetrievalService, ChromaVectorStore]:
    global _retrieval_service, _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(
            embedding_client=OllamaEmbeddingClient(),
            vector_store=_vector_store,
        )
    return _retrieval_service, _vector_store


def _normalize_identifier(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name).strip().lower()


def _match_source(product: str, sources: list[str]) -> str | None:
    if not product or not sources:
        return None
    normalized_product = _normalize_identifier(product)
    for source in sources:
        if _normalize_identifier(product) == _normalize_identifier(source):
            return source
    for source in sources:
        if normalized_product in _normalize_identifier(source):
            return source
    for source in sources:
        if _normalize_identifier(source).startswith(normalized_product):
            return source
    return None


def _derive_sources(chunks: list[RetrievedChunk]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    result: list[str] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page_number)
        if key not in seen:
            seen.add(key)
            result.append(f"{chunk.source} — page {chunk.page_number}")
    return result


async def analyse_problem(
    question: str,
    product: str,
    assistance_type: str = "auto",
    image: ProcessedImage | None = None,
) -> dict:
    logger.info(
        "Analysis requested — product=%s type=%s image=%s",
        product,
        assistance_type,
        f"{image.width}x{image.height}" if image else "none",
    )

    context: str | None = None
    sources: list[str] = []

    try:
        svc, store = _get_retrieval_components()
        available_sources = store.get_sources()
        matched_source = _match_source(product, available_sources) if available_sources else None
        chunks = await svc.retrieve(
            query=question,
            top_k=5,
            source=matched_source,
        )
        if chunks:
            context = RetrievalService.format_context(chunks, max_characters=4000)
            sources = _derive_sources(chunks)
            logger.info(
                "RAG context derived — chunks=%d sources=%s",
                len(chunks),
                sources,
            )
    except Exception as exc:
        logger.warning("RAG retrieval failed (continuing without context): %s", exc)

    result = await route_to_agent(
        question=question,
        product=product,
        assistance_type=assistance_type,
        context=context,
        image=image,
    )

    result["assistance_type"] = assistance_type
    result["sources"] = sources
    return result


async def route_request(
    question: str,
    product: str,
    assistance_type: str,
    image: ProcessedImage | None = None,
) -> dict:
    return await analyse_problem(
        question=question,
        product=product,
        assistance_type=assistance_type,
        image=image,
    )

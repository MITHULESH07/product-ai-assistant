"""RAG Pipeline Diagnostic Script.

Verifies the end-to-end retrieval pipeline without calling an LLM.

Usage:
    cd backend
    python scripts/test_rag_pipeline.py --product "MG90S" --question "What voltage?"

Options:
    --product       Product name (used for source filtering, optional)
    --question      Question to test retrieval against
    --top-k         Number of chunks to retrieve (default: 3)
    --min-relevance Minimum relevance threshold (default: 0.0)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_diagnostic")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test the RAG retrieval pipeline")
    parser.add_argument("--product", default=None, help="Product name for source filtering")
    parser.add_argument("--question", default="What is the recommended operating voltage?", help="Question to test")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--min-relevance", type=float, default=0.0, help="Minimum relevance threshold")
    args = parser.parse_args()

    from app.core.config import settings

    try:
        settings.validate()
    except RuntimeError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("RAG Pipeline Diagnostic")
    logger.info("=" * 60)
    logger.info("Product:       %s", args.product or "(none)")
    logger.info("Question:      %s", args.question)
    logger.info("Top K:         %d", args.top_k)
    logger.info("Min Relevance: %.2f", args.min_relevance)
    logger.info("Embed Model:   %s", settings.ollama_embedding_model)
    logger.info("Ollama URL:    %s", settings.ollama_base_url)
    logger.info("")
    logger.info("--- Step 1: List available sources ---")

    from app.rag.vector_store import ChromaVectorStore

    store = ChromaVectorStore()
    sources = store.get_sources()
    total_chunks = store.count()

    logger.info("Available sources (%d): %s", len(sources), sources)
    logger.info("Total chunks in collection: %d", total_chunks)

    if not sources:
        logger.warning("No documents indexed. Ingest a PDF first via POST /api/documents/ingest")
        logger.info("=" * 60)
        return

    logger.info("")
    logger.info("--- Step 2: Source matching ---")

    if args.product:
        from app.services.analysis_service import _match_source, _normalize_identifier

        matched = _match_source(args.product, sources)
        if matched:
            logger.info("Product '%s' matched to source: '%s'", args.product, matched)
        else:
            logger.info("No source match for product '%s'", args.product)
            logger.info("  normalized product: %s", _normalize_identifier(args.product))
            for s in sources:
                logger.info("  normalized source:  %s  (%s)", _normalize_identifier(s), s)

    logger.info("")
    logger.info("--- Step 3: Retrieve chunks ---")

    from app.rag.embedding_client import OllamaEmbeddingClient
    from app.rag.retrieval_service import RetrievalService

    embed_client = OllamaEmbeddingClient()
    svc = RetrievalService(
        embedding_client=embed_client,
        vector_store=store,
    )

    try:
        chunks = await svc.retrieve(
            query=args.question,
            top_k=args.top_k,
            source=matched if args.product else None,
            minimum_relevance=args.min_relevance,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        sys.exit(1)

    if not chunks:
        logger.warning("No chunks retrieved.")
        logger.info("")
        logger.info("--- Step 4: Formatted context ---")
        logger.info("(empty)")
    else:
        logger.info("Retrieved %d chunk(s):", len(chunks))
        logger.info("")

        for i, c in enumerate(chunks, 1):
            preview = c.text[:150].replace("\n", " ¶ ")
            logger.info(
                "  #%d  [%s]  src=%s  p=%s  dist=%.4f  relevance=%.4f",
                i,
                c.chunk_id,
                c.source,
                c.page_number,
                c.distance,
                c.relevance_score,
            )
            logger.info("       %s...", preview)
            logger.info("")

        logger.info("--- Step 4: Formatted context ---")
        formatted = svc.format_context(chunks, max_characters=4000)
        logger.info("Context length: %d characters", len(formatted))
        logger.info("")
        print(formatted)

    logger.info("")
    logger.info("--- Step 5: Derived sources ---")
    from app.services.analysis_service import _derive_sources

    derived = _derive_sources(chunks)
    logger.info("Sources that would appear in response: %s", derived)

    logger.info("=" * 60)
    logger.info("Diagnostic complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
import math
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.rag.text_chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk with its embedding vector."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    embedding: list[float]


class EmbeddingClientError(Exception):
    """Raised when an embedding operation fails."""


class OllamaEmbeddingClient:
    """Generates embedding vectors via a remote Ollama embeddings model.

    Args:
        max_concurrency: Maximum number of concurrent embedding
            requests to the remote server.
        max_retries: Number of retries for transient connection or
            timeout errors.
    """

    def __init__(
        self,
        max_concurrency: int = 2,
        max_retries: int = 1,
    ) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_embedding_model
        self._timeout = settings.ollama_timeout
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._embed_url = f"{self._base_url}/api/embed"

        logger.info(
            "OllamaEmbeddingClient initialized — model=%s max_concurrency=%d",
            self._model,
            self._max_concurrency,
        )

    # ── public API ──────────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: The text to embed. Must be non-empty after stripping.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingClientError: If the text is empty, the server
                cannot be reached, the response is malformed, or the
                embedding values are invalid.
        """
        if not text or not text.strip():
            raise EmbeddingClientError("Cannot embed empty or whitespace-only text")

        payload = {
            "model": self._model,
            "input": text.strip(),
        }

        data = await self._request_with_retry(payload)
        return self._extract_embedding(data)

    async def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Embed a single document chunk.

        Args:
            chunk: A ``DocumentChunk`` to embed.

        Returns:
            An ``EmbeddedChunk`` with the embedding vector.

        Raises:
            EmbeddingClientError: If the embedding fails.
        """
        embedding = await self.embed_text(chunk.text)
        return EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=embedding,
        )

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """Embed a list of document chunks with bounded concurrency.

        Args:
            chunks: The document chunks to embed.

        Returns:
            A list of ``EmbeddedChunk`` objects in the same order as
            the input.

        Raises:
            EmbeddingClientError: If any single chunk fails to embed.
                The error message includes the failed ``chunk_id``.
        """
        if not chunks:
            logger.info("embed_chunks received empty list — returning []")
            return []

        logger.info(
            "Embedding %d chunk(s) — model=%s concurrency=%d",
            len(chunks),
            self._model,
            self._max_concurrency,
        )

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def embed_one(chunk: DocumentChunk) -> EmbeddedChunk:
            async with semaphore:
                try:
                    return await self.embed_chunk(chunk)
                except EmbeddingClientError:
                    raise
                except Exception as exc:
                    raise EmbeddingClientError(
                        f"Unexpected error embedding chunk '{chunk.chunk_id}': {exc}"
                    ) from exc

        tasks = [embed_one(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embedded: list[EmbeddedChunk] = []
        for chunk, result in zip(chunks, results, strict=True):
            if isinstance(result, EmbeddingClientError):
                logger.error(
                    "Embedding failed — chunk_id=%s source=%s error=%s",
                    chunk.chunk_id,
                    chunk.source,
                    result,
                )
                raise EmbeddingClientError(
                    f"Embedding failed for chunk '{chunk.chunk_id}': {result}"
                ) from result
            if isinstance(result, Exception):
                logger.error(
                    "Embedding failed — chunk_id=%s source=%s error=%s",
                    chunk.chunk_id,
                    chunk.source,
                    result,
                )
                raise EmbeddingClientError(
                    f"Embedding failed for chunk '{chunk.chunk_id}': {result}"
                ) from result
            embedded.append(result)

        logger.info(
            "Embedding complete — %d chunk(s) embedded successfully",
            len(embedded),
        )
        return embedded

    # ── internal helpers ─────────────────────────────────────────────

    async def _request_with_retry(self, payload: dict) -> dict:
        """POST the embedding payload with optional retry on transient errors.

        Args:
            payload: The JSON body for the ``/api/embed`` request.

        Returns:
            The parsed JSON response dict.

        Raises:
            EmbeddingClientError: On connection, timeout, HTTP, or
                parsing errors.
        """
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._embed_url, json=payload)

                response.raise_for_status()
                return response.json()

            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning(
                    "Embedding connection error (attempt %d/%d) — %s",
                    attempt + 1,
                    1 + self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0)
                    continue
                raise EmbeddingClientError(
                    "Cannot connect to the remote Ollama server for embedding. "
                    "Ensure the teammate's laptop is powered on, "
                    "Ollama is running, and both laptops are on the same network."
                ) from exc

            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "Embedding timeout (attempt %d/%d) — %s",
                    attempt + 1,
                    1 + self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0)
                    continue
                raise EmbeddingClientError(
                    f"The embedding model did not respond within "
                    f"{self._timeout} seconds."
                ) from exc

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text[:500]

                if status == 404:
                    try:
                        error_data = exc.response.json()
                        ollama_msg = error_data.get("error", body)
                    except (ValueError, TypeError):
                        ollama_msg = body
                    raise EmbeddingClientError(
                        f"The configured embedding model '{self._model}' is not "
                        f"available on the remote server. Error: {ollama_msg}"
                    ) from exc

                raise EmbeddingClientError(
                    f"Embedding request returned HTTP {status}: {body}"
                ) from exc

            except (ValueError, TypeError) as exc:
                raise EmbeddingClientError(
                    f"Malformed response from embedding server: {exc}"
                ) from exc

        # Should not reach here
        raise EmbeddingClientError(
            f"Embedding request failed after {1 + self._max_retries} attempt(s)"
        )

    @staticmethod
    def _extract_embedding(data: dict) -> list[float]:
        """Validate and extract a single embedding vector from the API response.

        Args:
            data: The parsed JSON response from ``/api/embed``.

        Returns:
            A list of floats.

        Raises:
            EmbeddingClientError: If the response is missing the
                embedding field, the vector is empty, or any value is
                non-numeric or non-finite.
        """
        embeddings = data.get("embeddings")

        if embeddings is None:
            raise EmbeddingClientError(
                "Embedding response is missing the 'embeddings' field. "
                f"Keys: {list(data.keys())}"
            )

        if not isinstance(embeddings, list) or len(embeddings) == 0:
            raise EmbeddingClientError(
                "Embedding response contains an empty or non-list 'embeddings' field"
            )

        vector = embeddings[0]

        if not isinstance(vector, list):
            raise EmbeddingClientError(
                f"Expected embedding to be a list, got {type(vector).__name__}"
            )

        if len(vector) == 0:
            raise EmbeddingClientError(
                "Embedding response contains an empty embedding vector"
            )

        for i, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise EmbeddingClientError(
                    f"Non-numeric value at index {i}: {value!r} "
                    f"({type(value).__name__})"
                )
            if not math.isfinite(value):
                raise EmbeddingClientError(
                    f"Non-finite value at index {i}: {value!r}"
                )

        return [float(v) for v in vector]


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    from app.rag.document_loader import DocumentLoader
    from app.rag.text_chunker import TextChunker

    loader = DocumentLoader()
    try:
        pages = loader.load_pdf(pdf_path)
    except Exception as exc:
        print(f"Error loading PDF: {exc}")
        sys.exit(1)

    chunker = TextChunker()
    chunks = chunker.chunk_pages(pages)

    print(f"\nLoaded {len(pages)} page(s) → {len(chunks)} chunk(s)")
    if not chunks:
        print("No chunks to embed.")
        sys.exit(0)

    n = min(3, len(chunks))
    print(f"Embedding first {n} chunk(s)...\n")

    client = OllamaEmbeddingClient()

    async def run() -> None:
        for i in range(n):
            try:
                ec = await client.embed_chunk(chunks[i])
                print(
                    f"  [{ec.chunk_id}] "
                    f"src={ec.source} p={ec.page_number} "
                    f"text_len={len(ec.text)} "
                    f"dim={len(ec.embedding)} "
                    f"values={ec.embedding[:5]}..."
                )
            except EmbeddingClientError as exc:
                print(f"  [{chunks[i].chunk_id}] ERROR: {exc}")

    asyncio.run(run())

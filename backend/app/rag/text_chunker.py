import logging
from dataclasses import dataclass

from app.rag.document_loader import DocumentPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of text extracted from a single page of a document."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


class TextChunkerError(Exception):
    """Raised when the chunker is misconfigured."""


class TextChunker:
    """Splits document pages into paragraph-aware, overlapping chunks.

    Each page is processed independently; chunks never cross page
    boundaries.  Paragraphs (separated by ``\\n\\n``) are kept intact
    when possible.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters carried
            between consecutive chunks on the same page.

    Raises:
        TextChunkerError: If the configuration values are invalid.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise TextChunkerError(
                f"chunk_size must be greater than zero, got {chunk_size}"
            )
        if chunk_overlap < 0:
            raise TextChunkerError(
                f"chunk_overlap must be zero or greater, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise TextChunkerError(
                f"chunk_overlap ({chunk_overlap}) must be smaller "
                f"than chunk_size ({chunk_size})"
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── public API ──────────────────────────────────────────────────

    def chunk_pages(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """Convert page-wise text into overlapping, paragraph-aware chunks.

        Args:
            pages: A list of :class:`DocumentPage` objects, typically
                  returned by :class:`DocumentLoader.load_pdf`.

        Returns:
            A list of :class:`DocumentChunk` objects, ordered by page
            and then by chunk index.
        """
        logger.info("Chunking %d page(s)", len(pages))

        all_chunks: list[DocumentChunk] = []
        total_skipped = 0
        current_page_count = 0

        for page in pages:
            paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]

            if not paragraphs:
                total_skipped += 1
                continue

            logger.debug("Chunking source=%s page=%d", page.source, page.page_number)

            page_chunks = self._chunk_paragraphs(paragraphs)

            for idx, text in enumerate(page_chunks):
                chunk_id = self._make_chunk_id(
                    page.source, page.page_number, idx
                )
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=idx,
                    )
                )

            logger.debug(
                "Page %d → %d chunk(s)", page.page_number, len(page_chunks)
            )
            current_page_count += len(page_chunks)

        logger.info(
            "Chunking complete: total_pages=%d total_chunks=%d skipped=%d",
            len(pages),
            len(all_chunks),
            total_skipped,
        )

        return all_chunks

    # ── internal chunking logic ─────────────────────────────────────

    def _chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Build a list of chunk texts from a page's paragraphs.

        Args:
            paragraphs: Non-empty, stripped paragraph strings.

        Returns:
            A list of chunk text strings (stripped).
        """
        chunks: list[str] = []
        buffer: list[str] = []

        def flush_buffer() -> str | None:
            nonlocal buffer
            if not buffer:
                return None
            text = "\n\n".join(buffer).strip()
            buffer = []
            return text if text else None

        for para in paragraphs:
            para_len = len(para)

            # Paragraph too large — flush buffer, then split paragraph
            if para_len > self._chunk_size:
                prev = flush_buffer()
                if prev:
                    chunks.append(prev)
                for piece in self._split_long_paragraph(para):
                    chunks.append(piece)
                continue

            # Empty buffer — start with overlap context if available
            if not buffer:
                overlap = self._last_overlap(chunks[-1]) if chunks else ""
                if overlap:
                    candidate = overlap + "\n\n" + para
                    if len(candidate) <= self._chunk_size:
                        buffer = [candidate]
                        continue
                buffer = [para]
                continue

            # Try to append to existing buffer
            candidate = "\n\n".join(buffer + [para])
            if len(candidate) <= self._chunk_size:
                buffer.append(para)
            else:
                prev = flush_buffer()
                if prev:
                    chunks.append(prev)
                overlap = self._last_overlap(prev) if prev else ""
                if overlap:
                    buffer = [overlap + "\n\n" + para]
                else:
                    buffer = [para]

        last = flush_buffer()
        if last:
            chunks.append(last)

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """Split a single paragraph that exceeds *chunk_size*.

        Uses a sliding window with overlap.

        Args:
            paragraph: The over-sized paragraph string.

        Returns:
            A list of text pieces, each within *chunk_size*.
        """
        pieces: list[str] = []
        start = 0
        text_len = len(paragraph)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            if end < text_len:
                break_at = paragraph.rfind(" ", start + self._chunk_size // 2, end)
                if break_at > start:
                    end = break_at

            piece = paragraph[start:end].strip()
            if piece:
                pieces.append(piece)

            start = end
            if start < text_len:
                start = max(start - self._chunk_overlap, start)
            else:
                break

        return pieces

    def _last_overlap(self, previous_chunk: str | None) -> str:
        """Return the trailing *chunk_overlap* characters of a chunk.

        Args:
            previous_chunk: The text of the previous chunk, or ``None``.

        Returns:
            Up to *chunk_overlap* trailing characters, or an empty
            string.
        """
        if (
            previous_chunk is None
            or self._chunk_overlap <= 0
            or len(previous_chunk) <= self._chunk_overlap
        ):
            return ""
        return previous_chunk[-self._chunk_overlap :]

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_chunk_id(source: str, page_number: int, chunk_index: int) -> str:
        """Build a deterministic, human-readable chunk identifier.

        Format: ``<source>:p<page_number>:c<chunk_index>``

        Colons in *source* are replaced with ``-`` to avoid ambiguity.

        Args:
            source: The PDF filename.
            page_number: 1-based page number.
            chunk_index: 0-based chunk index on that page.

        Returns:
            A stable chunk ID string.
        """
        safe_source = source.replace(":", "-")
        return f"{safe_source}:p{page_number}:c{chunk_index}"


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    from app.rag.document_loader import DocumentLoader

    loader = DocumentLoader()
    try:
        pages = loader.load_pdf(pdf_path)
    except Exception as exc:
        print(f"Error loading PDF: {exc}")
        sys.exit(1)

    chunker = TextChunker()
    chunks = chunker.chunk_pages(pages)

    print(f"\nLoaded {len(pages)} page(s), produced {len(chunks)} chunk(s):\n")
    for c in chunks:
        preview = c.text[:200].replace("\n", " ¶ ")
        print(
            f"  [{c.chunk_id}] src={c.source} "
            f"p={c.page_number} c={c.chunk_index} "
            f"len={len(c.text)}\n"
            f"    {preview}...\n"
        )

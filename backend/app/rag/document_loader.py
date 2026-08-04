import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentPage:
    """A single page extracted from a PDF document."""

    page_number: int
    text: str
    source: str


class DocumentLoaderError(Exception):
    """Raised when a PDF cannot be loaded or read."""


class DocumentLoader:
    """Loads PDF documents and extracts text page by page.

    Handles file validation, encryption detection, text
    normalization, and empty-page skipping.
    """

    def load_pdf(
        self,
        pdf_path: str,
        source_name: str | None = None,
    ) -> list[DocumentPage]:
        """Load a PDF and return its pages as a list of DocumentPage.

        Args:
            pdf_path: Path to the PDF file.
            source_name: Optional override for the source name stored
                in each page. When ``None``, uses ``path.name``.

        Returns:
            A list of DocumentPage objects, one per non-empty page.

        Raises:
            DocumentLoaderError: If the file is missing, has a bad
                extension, is encrypted, or cannot be read.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(pdf_path)

        self._validate_pdf_path(path)

        source = source_name or path.name

        logger.info("Loading PDF: %s (source=%s)", path.name, source)

        try:
            reader = pypdf.PdfReader(path)
        except pypdf.errors.PdfReadError as exc:
            raise DocumentLoaderError(
                f"Cannot read PDF file '{path.name}': {exc}"
            ) from exc

        if reader.is_encrypted:
            raise DocumentLoaderError(
                f"PDF file '{path.name}' is encrypted and cannot be read"
            )

        total_pages = len(reader.pages)
        pages: list[DocumentPage] = []
        skipped = 0

        for i in range(total_pages):
            page = reader.pages[i]
            raw_text = page.extract_text() or ""
            text = self._normalize_text(raw_text)

            if not text:
                skipped += 1
                continue

            pages.append(
                DocumentPage(
                    page_number=i + 1,
                    text=text,
                    source=source,
                )
            )

        logger.info(
            "PDF loaded: %s | total_pages=%d extracted=%d skipped=%d",
            path.name,
            total_pages,
            len(pages),
            skipped,
        )

        return pages

    # ── private helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_pdf_path(path: Path) -> None:
        """Validate that *path* exists and has a ``.pdf`` extension.

        Args:
            path: The file path to validate.

        Raises:
            FileNotFoundError: If the file does not exist.
            DocumentLoaderError: If the extension is not ``.pdf``.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )
        if path.suffix.lower() != ".pdf":
            raise DocumentLoaderError(
                f"File '{path.name}' is not a PDF "
                f"(got extension '{path.suffix}')"
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace in extracted text.

        Steps performed in order:

        1. Normalize line endings (``\\r\\n`` → ``\\n``, ``\\r`` → ``\\n``)
        2. Replace tabs with a single space
        3. Collapse consecutive spaces into one
        4. Collapse three or more consecutive newlines into two
        5. Strip leading and trailing whitespace

        Args:
            text: Raw text extracted from a PDF page.

        Returns:
            Normalized text with consistent spacing, or an empty string.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", " ")
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    loader = DocumentLoader()

    try:
        pages = loader.load_pdf(pdf_path)
    except (DocumentLoaderError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"\nLoaded {len(pages)} page(s) from '{pdf_path}':\n")
    for page in pages:
        preview = page.text[:200].replace("\n", " ¶ ")
        print(f"  Page {page.page_number}: {preview}...\n")

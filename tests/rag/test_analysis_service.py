"""Unit tests for the analysis-service RAG integration layer.

Tests the helper functions that connect retrieval to the analysis pipeline:
``_normalize_identifier``, ``_match_source``, ``_derive_sources``.
"""

import pytest

from app.rag.retrieval_service import RetrievedChunk
from app.services.analysis_service import (
    _derive_sources,
    _match_source,
    _normalize_identifier,
)


class TestNormalizeIdentifier:
    def test_lowercases(self):
        assert _normalize_identifier("MG90S") == "mg90s"

    def test_removes_hyphens(self):
        assert _normalize_identifier("MG-90S") == "mg90s"

    def test_removes_underscores(self):
        assert _normalize_identifier("MG_90S") == "mg90s"

    def test_removes_spaces(self):
        assert _normalize_identifier("MG 90S") == "mg90s"

    def test_combined(self):
        assert _normalize_identifier("MG-90S Pro") == "mg90spro"

    def test_strips_whitespace(self):
        assert _normalize_identifier("  MG90S  ") == "mg90s"

    def test_empty_string(self):
        assert _normalize_identifier("") == ""

    def test_already_normalized(self):
        assert _normalize_identifier("esp32") == "esp32"

    def test_pdf_filename(self):
        assert _normalize_identifier("mg90s-manual.pdf") == "mg90smanual.pdf"


class TestMatchSource:
    def test_exact_match(self):
        sources = ["mg90s-manual.pdf", "esp32-datasheet.pdf"]
        assert _match_source("mg90s-manual.pdf", sources) == "mg90s-manual.pdf"

    def test_case_insensitive_match(self):
        sources = ["MG90S-Manual.pdf"]
        assert _match_source("mg90s-manual.pdf", sources) == "MG90S-Manual.pdf"

    def test_substring_match_product_in_source(self):
        sources = ["mg90s-manual.pdf", "esp32-datasheet.pdf"]
        assert _match_source("MG90S", sources) == "mg90s-manual.pdf"

    def test_no_match_returns_none(self):
        sources = ["esp32-datasheet.pdf"]
        assert _match_source("MG90S", sources) is None

    def test_empty_product_returns_none(self):
        sources = ["mg90s-manual.pdf"]
        assert _match_source("", sources) is None

    def test_empty_sources_returns_none(self):
        assert _match_source("MG90S", []) is None

    def test_hyphen_variation(self):
        sources = ["MG-90S-manual.pdf"]
        assert _match_source("MG90S", sources) == "MG-90S-manual.pdf"

    def test_space_variation(self):
        sources = ["mg 90s manual.pdf"]
        assert _match_source("MG90S", sources) == "mg 90s manual.pdf"

    def test_stem_match(self):
        sources = ["sg90-micro-servo.pdf"]
        assert _match_source("SG90", sources) == "sg90-micro-servo.pdf"

    def test_returns_first_match(self):
        sources = ["mg90s-v1.pdf", "mg90s-v2.pdf"]
        result = _match_source("MG90S", sources)
        assert result == "mg90s-v1.pdf"


class TestDeriveSources:
    def _chunk(self, source: str, page: int, chunk_id: str = "c0") -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            text="dummy",
            source=source,
            page_number=page,
            chunk_index=0,
            distance=0.1,
            relevance_score=0.9,
        )

    def test_single_chunk(self):
        chunks = [self._chunk("manual.pdf", 3)]
        assert _derive_sources(chunks) == ["manual.pdf — page 3"]

    def test_multiple_sources(self):
        chunks = [
            self._chunk("manual.pdf", 3),
            self._chunk("datasheet.pdf", 1),
        ]
        assert _derive_sources(chunks) == [
            "manual.pdf — page 3",
            "datasheet.pdf — page 1",
        ]

    def test_duplicate_source_page_deduplicated(self):
        chunks = [
            self._chunk("manual.pdf", 3, "c0"),
            self._chunk("manual.pdf", 3, "c1"),  # same source + page
            self._chunk("manual.pdf", 4, "c2"),
        ]
        assert _derive_sources(chunks) == [
            "manual.pdf — page 3",
            "manual.pdf — page 4",
        ]

    def test_preserves_order(self):
        chunks = [
            self._chunk("b.pdf", 1),
            self._chunk("a.pdf", 1),
            self._chunk("c.pdf", 1),
        ]
        result = _derive_sources(chunks)
        assert result == [
            "b.pdf — page 1",
            "a.pdf — page 1",
            "c.pdf — page 1",
        ]

    def test_empty_chunks(self):
        assert _derive_sources([]) == []

    def test_different_pages_same_source(self):
        chunks = [
            self._chunk("manual.pdf", 1),
            self._chunk("manual.pdf", 2),
            self._chunk("manual.pdf", 5),
        ]
        assert _derive_sources(chunks) == [
            "manual.pdf — page 1",
            "manual.pdf — page 2",
            "manual.pdf — page 5",
        ]

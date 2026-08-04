"""End-to-end integration tests for the RAG analysis pipeline.

Verifies that the RAG retrieval context is properly fetched, formatted,
and passed to the LLM, and that sources appear in the API response.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.rag.retrieval_service import RetrievedChunk

pytestmark = pytest.mark.asyncio

client = TestClient(app)

_LLM_TARGET = "app.clients.llm_factory.generate_assistance"
_RAG_TARGET = "app.services.analysis_service._get_retrieval_components"


def _make_chunk(
    chunk_id: str = "manual.pdf:p1:c0",
    text: str = "The MG90S servo operates at 4.8-6.0V.",
    source: str = "manual.pdf",
    page_number: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=0,
        distance=0.1,
        relevance_score=0.9,
    )


class TestRagE2E:
    _captured_llm_kwargs: dict = {}

    def _llm_side_effect(self, **kwargs):
        self._captured_llm_kwargs = kwargs
        return {
            "intent": kwargs.get("assistance_type", "troubleshooting"),
            "product": kwargs.get("product", ""),
            "summary": "Mock analysis summary",
            "possible_causes": ["Mock cause 1"],
            "steps": ["Mock step 1"],
            "warning": "",
            "escalation_required": False,
            "sources": [],
        }

    async def _run_analyse(self, *,
                            question: str = "Why is my servo overheating?",
                            product: str = "MG90S Servo",
                            assistance_type: str = "troubleshooting",
                            image_bytes: bytes | None = None):
        mock_store = MagicMock()
        mock_store.get_sources.return_value = self._mock_sources
        mock_svc = AsyncMock()
        mock_svc.retrieve = AsyncMock(return_value=self._mock_chunks)

        with patch(_RAG_TARGET, return_value=(mock_svc, mock_store)):
            with patch(_LLM_TARGET) as mock_llm:
                mock_llm.side_effect = self._llm_side_effect

                if image_bytes:
                    response = client.post(
                        "/api/analyse",
                        data={
                            "question": question,
                            "product": product,
                            "assistance_type": assistance_type,
                        },
                        files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
                    )
                else:
                    response = client.post(
                        "/api/analyse",
                        json={
                            "question": question,
                            "product": product,
                            "assistance_type": assistance_type,
                        },
                    )
                return response, mock_svc

    def setup_method(self):
        self._mock_sources = ["manual.pdf"]
        self._mock_chunks = []

    # ── Happy path: RAG context is passed to the LLM ──────────────

    async def test_rag_context_included_when_chunks_found(self):
        self._mock_chunks = [
            _make_chunk(text="The MG90S servo operates at 4.8-6.0V.", page_number=3),
            _make_chunk(chunk_id="manual.pdf:p2:c0", text="Do not exceed 6.0V.", page_number=4),
        ]
        response, _ = await self._run_analyse(
            question="What voltage for MG90S?",
            product="MG90S Servo",
        )
        assert response.status_code == 200
        data = response.json()
        assert "[Source: manual.pdf, Page: 3]" in self._captured_llm_kwargs.get("context", "")
        assert "4.8-6.0V" in self._captured_llm_kwargs.get("context", "")
        assert data["sources"] == ["manual.pdf — page 3", "manual.pdf — page 4"]

    async def test_rag_context_included_for_operation_type(self):
        self._mock_chunks = [
            _make_chunk(text="Connect brown wire to ground.", page_number=5),
        ]
        response, _ = await self._run_analyse(
            question="How to wire the servo?",
            product="MG90S Servo",
            assistance_type="operation",
        )
        assert response.status_code == 200
        assert "brown wire" in self._captured_llm_kwargs.get("context", "")
        assert self._captured_llm_kwargs.get("assistance_type") == "operation"

    async def test_rag_context_included_for_maintenance_type(self):
        self._mock_chunks = [
            _make_chunk(text="Clean servo every 6 months.", page_number=10),
        ]
        response, _ = await self._run_analyse(
            question="How to maintain the servo?",
            product="MG90S Servo",
            assistance_type="maintenance",
        )
        assert response.status_code == 200
        assert "Clean servo" in self._captured_llm_kwargs.get("context", "")
        assert self._captured_llm_kwargs.get("assistance_type") == "maintenance"

    async def test_source_matching_uses_product_name(self):
        self._mock_sources = ["mg90s-servo-manual.pdf", "esp32-datasheet.pdf"]
        self._mock_chunks = [
            _make_chunk(source="mg90s-servo-manual.pdf", text="Torque: 1.8 kg-cm", page_number=7),
        ]
        response, _ = await self._run_analyse(
            question="What is the torque?",
            product="MG90S Servo",
        )
        assert response.status_code == 200
        assert "Torque" in self._captured_llm_kwargs.get("context", "")
        assert response.json()["sources"] == ["mg90s-servo-manual.pdf — page 7"]

    # ── No chunks found ─────────────────────────────────────────

    async def test_no_chunks_still_succeeds_without_context(self):
        self._mock_chunks = []
        response, _ = await self._run_analyse(
            question="Something not in the manual",
            product="ESP32",
        )
        assert response.status_code == 200
        assert self._captured_llm_kwargs.get("context") is None
        assert response.json()["sources"] == []

    async def test_no_matching_source_still_succeeds(self):
        self._mock_sources = ["esp32-datasheet.pdf"]
        self._mock_chunks = []
        response, _ = await self._run_analyse(
            question="How do I fix this?",
            product="MG90S Servo",
        )
        assert response.status_code == 200
        assert response.json()["sources"] == []

    async def test_empty_sources_list_still_succeeds(self):
        self._mock_sources = []
        self._mock_chunks = []
        response, _ = await self._run_analyse(
            question="Test question",
            product="ESP32",
        )
        assert response.status_code == 200
        assert response.json()["sources"] == []

    # ── RAG failure gracefully degraded ──────────────────────────

    async def test_retrieval_failure_graceful_degradation(self):
        """When RAG retrieval throws, the analysis should still complete."""
        with patch(_RAG_TARGET, side_effect=RuntimeError("ChromaDB unavailable")):
            with patch(_LLM_TARGET) as mock_llm:
                mock_llm.side_effect = self._llm_side_effect
                response = client.post(
                    "/api/analyse",
                    json={
                        "question": "Is my servo broken?",
                        "product": "MG90S Servo",
                        "assistance_type": "troubleshooting",
                    },
                )
        assert response.status_code == 200
        assert self._captured_llm_kwargs.get("context") is None
        assert response.json()["sources"] == []

    # ── Image + RAG context together ─────────────────────────────

    async def test_image_and_rag_context_together(self):
        self._mock_chunks = [
            _make_chunk(text="The MG90S has plastic gears.", page_number=2),
        ]
        buf = io.BytesIO()
        img = Image.new("RGB", (64, 64), color="red")
        img.save(buf, format="JPEG")
        buf.seek(0)

        response, _ = await self._run_analyse(
            question="What type of gears?",
            product="MG90S Servo",
            image_bytes=buf.getvalue(),
        )
        assert response.status_code == 200
        assert "plastic gears" in self._captured_llm_kwargs.get("context", "")
        assert self._captured_llm_kwargs.get("image") is not None
        assert response.json()["sources"] == ["manual.pdf — page 2"]

    # ── Provider switching (NVIDIA) ─────────────────────────────

    async def test_nvidia_provider_rag_context(self):
        import json

        self._mock_chunks = [
            _make_chunk(text="Operating voltage: 4.8-6.0V", page_number=3),
        ]
        mock_store = MagicMock()
        mock_store.get_sources.return_value = self._mock_sources
        mock_svc = AsyncMock()
        mock_svc.retrieve = AsyncMock(return_value=self._mock_chunks)

        with patch(_RAG_TARGET, return_value=(mock_svc, mock_store)):
            with patch("app.clients.llm_factory._resolve_provider", return_value="nvidia"):
                with patch("app.clients.nvidia_client.httpx.AsyncClient") as mock_httpx:
                    mock_resp = MagicMock()
                    mock_resp.status_code = 200
                    mock_resp.text = json.dumps({
                        "choices": [{"message": {"content": json.dumps({
                            "intent": "troubleshooting",
                            "product": "MG90S Servo",
                            "summary": "Mock analysis",
                            "possible_causes": [],
                            "steps": [],
                            "warning": "",
                            "escalation_required": False,
                            "sources": [],
                        })}}]
                    })
                    mock_resp.json.return_value = json.loads(mock_resp.text)
                    mock_httpx.return_value.__aenter__.return_value.post.return_value = mock_resp

                    response = client.post(
                        "/api/analyse",
                        json={
                            "question": "What voltage?",
                            "product": "MG90S Servo",
                            "assistance_type": "troubleshooting",
                        },
                    )
        assert response.status_code == 200
        assert "sources" in response.json()

    # ── Context formatting truncation ────────────────────────────

    async def test_large_context_truncated(self):
        self._mock_chunks = [
            _make_chunk(text="A" * 5000, page_number=1),
            _make_chunk(chunk_id="b", text="B" * 5000, page_number=2),
        ]
        response, _ = await self._run_analyse(
            question="Test truncation",
            product="MG90S Servo",
        )
        assert response.status_code == 200
        context = self._captured_llm_kwargs.get("context", "")
        assert len(context) <= 4000

    # ── Validates product used for source matching ────────────────

    async def test_source_filter_passed_to_retrieval(self):
        mock_store = MagicMock()
        mock_store.get_sources.return_value = ["mg90s-servo-manual.pdf"]
        mock_svc = AsyncMock()
        mock_svc.retrieve = AsyncMock(return_value=[
            _make_chunk(source="mg90s-servo-manual.pdf", text="Some text", page_number=1),
        ])

        with patch(_RAG_TARGET, return_value=(mock_svc, mock_store)):
            with patch(_LLM_TARGET) as mock_llm:
                mock_llm.side_effect = self._llm_side_effect
                client.post(
                    "/api/analyse",
                    json={
                        "question": "Test",
                        "product": "MG90S Servo",
                        "assistance_type": "troubleshooting",
                    },
                )

        assert mock_svc.retrieve.await_count == 1
        _, kwargs = mock_svc.retrieve.await_args
        assert kwargs["source"] == "mg90s-servo-manual.pdf"

    async def test_no_source_filter_when_no_match(self):
        mock_store = MagicMock()
        mock_store.get_sources.return_value = ["esp32-datasheet.pdf"]
        mock_svc = AsyncMock()
        mock_svc.retrieve = AsyncMock(return_value=[])

        with patch(_RAG_TARGET, return_value=(mock_svc, mock_store)):
            with patch(_LLM_TARGET) as mock_llm:
                mock_llm.side_effect = self._llm_side_effect
                client.post(
                    "/api/analyse",
                    json={
                        "question": "Test",
                        "product": "MG90S Servo",
                        "assistance_type": "troubleshooting",
                    },
                )

        assert mock_svc.retrieve.await_count == 1
        _, kwargs = mock_svc.retrieve.await_args
        assert kwargs["source"] is None

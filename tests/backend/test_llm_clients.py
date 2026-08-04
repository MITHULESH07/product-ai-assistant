import json
import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

os.environ.setdefault("OLLAMA_BASE_URL", "http://test:11434")
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ["OLLAMA_VISION_MODEL"] = "test-ollama-vision-model"
os.environ.setdefault("NVIDIA_BASE_URL", "https://test.nvidia.ai")
os.environ.setdefault("NVIDIA_MODEL", "test-nvidia-model")
os.environ.setdefault("NVIDIA_VISION_MODEL", "test-nvidia-vision-model")

from app.core.config import settings

try:
    settings.validate()
except RuntimeError:
    pass

from app.clients.exceptions import LLMClientError
from app.clients.nvidia_client import (
    NvidiaClientError,
    extract_json,
    generate_assistance,
    generate_assistance_with_image,
)
from app.clients.ollama_client import (
    OllamaClientError,
    extract_json as ollama_extract_json,
    generate_assistance as ollama_generate_assistance,
    generate_assistance_with_image as ollama_generate_assistance_with_image,
)
from app.services.image_service import ProcessedImage
import app.clients.llm_factory as llm_factory_mod


@pytest.fixture(autouse=True)
def _reset_factory_cache():
    llm_factory_mod._generate_fn = None
    llm_factory_mod._vision_generate_fn = None
    yield
    llm_factory_mod._generate_fn = None
    llm_factory_mod._vision_generate_fn = None

SAMPLE_NVIDIA_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"summary": "test", "possible_causes": ["a"], "steps": ["b"]}'
            }
        }
    ]
}

SAMPLE_ANALYSIS = {"summary": "test", "possible_causes": ["a"], "steps": ["b"]}

SAMPLE_OLLAMA_RESPONSE = {"response": '{"summary": "test", "possible_causes": ["a"], "steps": ["b"]}'}


def _mock_httpx_client(
    data: dict | None = None,
    status_code: int = 200,
    side_effect: Exception | None = None,
) -> AsyncMock:
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.text = json.dumps(data) if data else ""
    if data is not None:
        mock_response.json.return_value = data
    if side_effect:
        mock_response.json.side_effect = side_effect

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    if side_effect is None:
        mock_client.post.return_value = mock_response
    else:
        mock_client.post.side_effect = side_effect
    return mock_client

# Individual async tests will use @pytest.mark.asyncio


class TestNvidiaExtractJson:
    def test_extract_plain_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_with_code_fence_json(self):
        result = extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_with_code_fence(self):
        result = extract_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_nested_json(self):
        result = extract_json('{"a": {"b": ["x", "y"]}}')
        assert result == {"a": {"b": ["x", "y"]}}

    def test_extract_finds_braces_in_noisy_text(self):
        text = "Some text before\n{\"key\": \"value\"}\nSome text after"
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_extract_no_braces_returns_summary(self):
        result = extract_json("not json at all")
        assert result["summary"] == "not json at all"
        assert result["possible_causes"] == []

    def test_extract_bad_braces_returns_summary(self):
        result = extract_json("{bad json}")
        assert "bad json" in result["summary"]
        assert result["possible_causes"] == []


@pytest.mark.asyncio
class TestNvidiaGenerateAssistance:
    PATCH_TARGET = "app.clients.nvidia_client.httpx.AsyncClient"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_successful_generation(self):
        mock_client = _mock_httpx_client(SAMPLE_NVIDIA_RESPONSE)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await generate_assistance(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
            )
        assert result == SAMPLE_ANALYSIS

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_includes_context(self):
        mock_client = _mock_httpx_client(SAMPLE_NVIDIA_RESPONSE)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await generate_assistance(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                context="some RAG context",
            )
        assert result == SAMPLE_ANALYSIS

    async def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(NvidiaClientError, match="NVIDIA_API_KEY is not set"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_401_raises_auth_error(self):
        mock_client = _mock_httpx_client(status_code=401)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="authentication failed"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_403_raises_access_denied(self):
        mock_client = _mock_httpx_client(status_code=403)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="access denied"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_404_raises_model_not_found(self):
        mock_client = _mock_httpx_client(status_code=404)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="not found"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_429_raises_rate_limit(self):
        mock_client = _mock_httpx_client(status_code=429)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="rate limit"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_5xx_raises_http_error(self):
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "502 Bad Gateway", request=Mock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="HTTP 502"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_connection_error_raises(self):
        mock_client = _mock_httpx_client(
            side_effect=httpx.ConnectError("connection refused")
        )
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="Cannot connect"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_timeout_raises(self):
        mock_client = _mock_httpx_client(
            side_effect=httpx.TimeoutException("timed out")
        )
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="did not respond within"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_missing_choices_raises(self):
        mock_client = _mock_httpx_client({"choices": []})
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="missing choices"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_empty_content_raises(self):
        mock_client = _mock_httpx_client({
            "choices": [{"message": {"content": ""}}]
        })
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="empty response"):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_nvidia_client_error_is_llm_client_error(self):
        mock_client = _mock_httpx_client(status_code=401)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(LLMClientError):
                await generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )


@pytest.mark.asyncio
class TestNvidiaGenerateAssistanceWithImage:
    PATCH_TARGET = "app.clients.nvidia_client.httpx.AsyncClient"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_successful_with_image(self):
        mock_client = _mock_httpx_client(SAMPLE_NVIDIA_RESPONSE)
        image = ProcessedImage(data=b"fake-image", media_type="image/jpeg", width=200, height=150)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await generate_assistance_with_image(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                image=image,
            )
        assert result == SAMPLE_ANALYSIS

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_image_none_raises(self):
        with pytest.raises(NvidiaClientError, match="called without an image"):
            await generate_assistance_with_image(
                question="test",
                product="test",
                assistance_type="troubleshooting",
                image=None,
            )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_vision_model_not_configured(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        with patch("app.clients.nvidia_client.settings") as mock_settings:
            mock_settings.nvidia_vision_model = None
            with pytest.raises(NvidiaClientError, match="NVIDIA_VISION_MODEL is not configured"):
                await generate_assistance_with_image(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                    image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_payload_format(self):
        image = ProcessedImage(data=b"fake-image-data", media_type="image/jpeg", width=200, height=150)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_NVIDIA_RESPONSE
        mock_client.post.return_value = mock_resp

        with patch(self.PATCH_TARGET, return_value=mock_client):
            await generate_assistance_with_image(
                question="q",
                product="p",
                assistance_type="operation",
                image=image,
            )

        call_kwargs = mock_client.post.call_args.kwargs
        sent = call_kwargs["json"]
        assert sent["model"] == "test-nvidia-vision-model"
        assert sent["messages"][0]["role"] == "system"
        assert sent["messages"][1]["role"] == "user"
        assert isinstance(sent["messages"][1]["content"], list)
        assert len(sent["messages"][1]["content"]) == 2
        assert sent["messages"][1]["content"][0]["type"] == "text"
        assert sent["messages"][1]["content"][1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in sent["messages"][1]["content"][1]["image_url"]["url"]

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_401_raises_auth_error(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(status_code=401)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="authentication failed"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_403_raises_access_denied(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(status_code=403)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="access denied"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_404_raises_model_not_found(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(status_code=404)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="not found"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_429_raises_rate_limit(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(status_code=429)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="rate limit"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_5xx_raises_http_error(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "502 Bad Gateway", request=Mock(), response=mock_response
        )
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="HTTP 502"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_connection_error_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("connection refused"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="Cannot connect"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_timeout_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(side_effect=httpx.TimeoutException("timed out"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="did not respond within"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_missing_choices_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client({"choices": []})
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="missing choices"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_empty_content_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client({"choices": [{"message": {"content": ""}}]})
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(NvidiaClientError, match="empty response"):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_nvidia_client_error_is_llm_client_error(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(status_code=401)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(LLMClientError):
                await generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
    async def test_includes_context(self):
        mock_client = _mock_httpx_client(SAMPLE_NVIDIA_RESPONSE)
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=200, height=150)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await generate_assistance_with_image(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                context="some RAG context",
                image=image,
            )
        assert result == SAMPLE_ANALYSIS


@patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
class TestOllamaExtractJson:
    def test_extract_plain_json(self):
        result = ollama_extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extract_with_code_fence_json(self):
        result = ollama_extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_with_code_fence(self):
        result = ollama_extract_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_extract_nested_json(self):
        result = ollama_extract_json('{"a": {"b": ["x", "y"]}}')
        assert result == {"a": {"b": ["x", "y"]}}

    def test_extract_finds_braces_in_noisy_text(self):
        text = "Some text before\n{\"key\": \"value\"}\nSome text after"
        result = ollama_extract_json(text)
        assert result == {"key": "value"}

    def test_extract_no_braces_returns_summary(self):
        result = ollama_extract_json("not json at all")
        assert result["summary"] == "not json at all"
        assert result["possible_causes"] == []

    def test_extract_bad_braces_returns_summary(self):
        result = ollama_extract_json("{bad json}")
        assert "bad json" in result["summary"]
        assert result["possible_causes"] == []


@patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://test:11434", "OLLAMA_MODEL": "test-model"}, clear=False)
@pytest.mark.asyncio
class TestOllamaGenerateAssistance:
    PATCH_TARGET = "app.clients.ollama_client.httpx.AsyncClient"

    async def test_successful_generation(self):
        mock_client = _mock_httpx_client(SAMPLE_OLLAMA_RESPONSE)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
            )
        assert result == SAMPLE_ANALYSIS

    async def test_includes_context(self):
        mock_client = _mock_httpx_client(SAMPLE_OLLAMA_RESPONSE)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                context="some RAG context",
            )
        assert result == SAMPLE_ANALYSIS

    async def test_connection_error_raises(self):
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("connection refused"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="Cannot connect"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_timeout_raises(self):
        mock_client = _mock_httpx_client(side_effect=httpx.TimeoutException("timed out"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="did not respond within"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_404_raises_model_not_found(self):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = '{"error": "model not found"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="not available"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_5xx_raises_http_error(self):
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "502 Bad Gateway", request=Mock(), response=mock_response
        )
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="HTTP 502"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_empty_response_raises(self):
        mock_client = _mock_httpx_client({"response": ""})
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="empty response"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_ollama_client_error_is_llm_client_error(self):
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("connection refused"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(LLMClientError):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )

    async def test_transient_400_retries_then_succeeds(self):
        mock_400 = Mock()
        mock_400.status_code = 400
        mock_400.text = '{"error": "model not loaded"}'
        mock_400.json.return_value = {"error": "model not loaded"}

        mock_200 = Mock()
        mock_200.status_code = 200
        mock_200.text = json.dumps(SAMPLE_OLLAMA_RESPONSE)
        mock_200.json.return_value = SAMPLE_OLLAMA_RESPONSE

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.side_effect = [mock_400, mock_200]

        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
            )
        assert result == SAMPLE_ANALYSIS
        assert mock_client.post.call_count == 2

    async def test_persistent_400_raises_http_error(self):
        mock_400 = Mock()
        mock_400.status_code = 400
        mock_400.text = '{"error": "bad request"}'
        mock_400.json.return_value = {"error": "bad request"}
        mock_400.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=Mock(), response=mock_400
        )

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_400

        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="HTTP 400"):
                await ollama_generate_assistance(
                    question="test", product="test", assistance_type="troubleshooting",
                )
        assert mock_client.post.call_count == 3


@pytest.mark.asyncio
class TestOllamaGenerateAssistanceWithImage:
    PATCH_TARGET = "app.clients.ollama_client.httpx.AsyncClient"

    async def test_successful_with_image(self):
        mock_client = _mock_httpx_client(SAMPLE_OLLAMA_RESPONSE)
        image = ProcessedImage(data=b"fake-image", media_type="image/jpeg", width=200, height=150)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance_with_image(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                image=image,
            )
        assert result == SAMPLE_ANALYSIS

    async def test_image_none_raises(self):
        with pytest.raises(OllamaClientError, match="called without an image"):
            await ollama_generate_assistance_with_image(
                question="test",
                product="test",
                assistance_type="troubleshooting",
                image=None,
            )

    async def test_vision_model_not_configured(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        with patch("app.clients.ollama_client.settings") as mock_settings:
            mock_settings.ollama_vision_model = None
            with pytest.raises(OllamaClientError, match="OLLAMA_VISION_MODEL is not configured"):
                await ollama_generate_assistance_with_image(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                    image=image,
                )

    async def test_payload_format(self):
        image = ProcessedImage(data=b"fake-image-data", media_type="image/jpeg", width=200, height=150)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(SAMPLE_OLLAMA_RESPONSE)
        mock_resp.json.return_value = SAMPLE_OLLAMA_RESPONSE
        mock_client.post.return_value = mock_resp

        with patch(self.PATCH_TARGET, return_value=mock_client):
            await ollama_generate_assistance_with_image(
                question="q",
                product="p",
                assistance_type="operation",
                image=image,
            )

        call_kwargs = mock_client.post.call_args.kwargs
        sent = call_kwargs["json"]
        assert sent["model"] == "test-ollama-vision-model"
        assert "images" in sent
        assert isinstance(sent["images"], list)
        assert len(sent["images"]) == 1
        import base64
        assert sent["images"][0] == base64.b64encode(image.data).decode("ascii")
        assert "prompt" in sent

    async def test_connection_error_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("connection refused"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="Cannot connect"):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_timeout_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(side_effect=httpx.TimeoutException("timed out"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="did not respond within"):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_404_raises_model_not_found(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = '{"error": "model not found"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=Mock(), response=mock_response
        )
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="not available"):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_5xx_raises_http_error(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "502 Bad Gateway", request=Mock(), response=mock_response
        )
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="HTTP 502"):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_empty_response_raises(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client({"response": ""})
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(OllamaClientError, match="empty response"):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_ollama_client_error_is_llm_client_error(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("connection refused"))
        with patch(self.PATCH_TARGET, return_value=mock_client):
            with pytest.raises(LLMClientError):
                await ollama_generate_assistance_with_image(
                    question="test", product="test", assistance_type="troubleshooting", image=image,
                )

    async def test_includes_context(self):
        mock_client = _mock_httpx_client(SAMPLE_OLLAMA_RESPONSE)
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=200, height=150)
        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance_with_image(
                question="test problem",
                product="test product",
                assistance_type="troubleshooting",
                context="some RAG context",
                image=image,
            )
        assert result == SAMPLE_ANALYSIS

    async def test_narrative_is_structured_via_text_model(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        narrative = "The image shows an ESP32 board with a red arrow pointing at GPIO0."
        structured = {
            "intent": "troubleshooting",
            "product": "test product",
            "summary": "GPIO0 wiring issue detected",
            "possible_causes": ["GPIO0 wired to GND"],
            "steps": ["Check the wiring at GPIO0"],
            "warning": None,
            "escalation_required": False,
        }

        mock_resp_1 = Mock()
        mock_resp_1.status_code = 200
        mock_resp_1.text = "{}"
        mock_resp_1.json.return_value = {"response": narrative}

        mock_resp_2 = Mock()
        mock_resp_2.status_code = 200
        mock_resp_2.text = "{}"
        mock_resp_2.json.return_value = {"response": json.dumps(structured)}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.side_effect = [mock_resp_1, mock_resp_2]

        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance_with_image(
                question="Why is my board not booting?",
                product="test product",
                assistance_type="troubleshooting",
                context="some RAG context",
                image=image,
            )

        assert result["summary"] == "GPIO0 wiring issue detected"
        assert result["possible_causes"] == ["GPIO0 wired to GND"]
        assert result["steps"] == ["Check the wiring at GPIO0"]
        assert mock_client.post.call_count == 2

        first_payload = mock_client.post.call_args_list[0].kwargs["json"]
        second_payload = mock_client.post.call_args_list[1].kwargs["json"]
        assert first_payload["model"] == "test-ollama-vision-model"
        assert "images" in first_payload
        assert "some RAG context" not in first_payload["prompt"]
        assert second_payload["model"] == settings.ollama_model
        assert second_payload["model"] != first_payload["model"]
        assert "images" not in second_payload
        assert "some RAG context" in second_payload["prompt"]

    async def test_narrative_structuring_failure_falls_back(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=1, height=1)
        narrative = "The image shows an ESP32 board."

        mock_resp_1 = Mock()
        mock_resp_1.status_code = 200
        mock_resp_1.text = "{}"
        mock_resp_1.json.return_value = {"response": narrative}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.side_effect = [mock_resp_1, httpx.ConnectError("connection refused")]

        with patch(self.PATCH_TARGET, return_value=mock_client):
            result = await ollama_generate_assistance_with_image(
                question="test",
                product="test product",
                assistance_type="troubleshooting",
                image=image,
            )

        assert result["summary"] == narrative
        assert result["possible_causes"] == []
        assert result["steps"] == []


@patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-12345"}, clear=False)
@pytest.mark.asyncio
class TestFactoryGenerateAssistance:
    async def test_factory_dispatches_to_ollama(self):
        with patch("app.clients.llm_factory._resolve_provider", return_value="ollama"):
            with patch("app.clients.ollama_client.httpx.AsyncClient") as mock_httpx:
                mock_resp = Mock()
                mock_resp.status_code = 200
                mock_resp.text = ""
                mock_resp.json.return_value = {"response": '{"summary": "test"}'}
                mock_client = AsyncMock()
                mock_client.__aenter__.return_value = mock_client
                mock_client.post.return_value = mock_resp
                mock_httpx.return_value = mock_client

                result = await llm_factory_mod.generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )
                assert result == {"summary": "test"}

    async def test_factory_dispatches_to_nvidia(self):
        with patch("app.clients.llm_factory._resolve_provider", return_value="nvidia"):
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
                with patch("app.clients.nvidia_client.httpx.AsyncClient") as mock_httpx:
                    mock_resp = Mock()
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = {
                        "choices": [{"message": {"content": '{"summary": "test"}'}}]
                    }
                    mock_client = AsyncMock()
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.post.return_value = mock_resp
                    mock_httpx.return_value = mock_client

                    result = await llm_factory_mod.generate_assistance(
                        question="test",
                        product="test",
                        assistance_type="troubleshooting",
                    )
                    assert result == {"summary": "test"}

    async def test_factory_caches_provider(self):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = {"response": '{"ok": true}'}

        ollama_client_mock = AsyncMock()
        ollama_client_mock.__aenter__.return_value = ollama_client_mock
        ollama_client_mock.post.return_value = mock_resp

        with patch("app.clients.llm_factory._resolve_provider", return_value="ollama"):
            with patch("app.clients.ollama_client.httpx.AsyncClient", return_value=ollama_client_mock):
                r1 = await llm_factory_mod.generate_assistance(
                    question="t", product="t", assistance_type="troubleshooting"
                )
                r2 = await llm_factory_mod.generate_assistance(
                    question="t", product="t", assistance_type="troubleshooting"
                )

        assert r1 == {"ok": True}
        assert r2 == {"ok": True}
        assert ollama_client_mock.post.call_count == 2

    async def test_factory_unsupported_provider_raises(self):
        with patch("app.clients.llm_factory._resolve_provider", side_effect=ValueError(
            "Unsupported LLM provider 'invalid'. Supported values: 'ollama', 'nvidia'."
        )):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                await llm_factory_mod.generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                )

    async def test_factory_dispatches_to_nvidia_vision(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=100, height=100)
        with patch("app.clients.llm_factory._resolve_provider", return_value="nvidia"):
            with patch("app.clients.nvidia_client.generate_assistance_with_image") as mock_vision_fn:
                mock_vision_fn.return_value = {"summary": "vision result"}
                result = await llm_factory_mod.generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="troubleshooting",
                    image=image,
                )
        assert result == {"summary": "vision result"}
        mock_vision_fn.assert_awaited_once_with(
            question="test",
            product="test",
            assistance_type="troubleshooting",
            context=None,
            image=image,
        )

    async def test_factory_dispatches_to_ollama_vision(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=100, height=100)
        with patch("app.clients.llm_factory._resolve_provider", return_value="ollama"):
            with patch("app.clients.ollama_client.generate_assistance_with_image") as mock_vision_fn:
                mock_vision_fn.return_value = {"summary": "vision result"}
                result = await llm_factory_mod.generate_assistance(
                    question="test",
                    product="test",
                    assistance_type="maintenance",
                    image=image,
                )
        assert result == {"summary": "vision result"}
        mock_vision_fn.assert_awaited_once_with(
            question="test",
            product="test",
            assistance_type="maintenance",
            context=None,
            image=image,
        )

    async def test_factory_vision_caches_separately(self):
        image = ProcessedImage(data=b"fake", media_type="image/jpeg", width=100, height=100)
        with patch("app.clients.llm_factory._resolve_provider", return_value="ollama"):
            with patch("app.clients.ollama_client.generate_assistance_with_image") as mock_vision:
                mock_vision.return_value = {"ok": True}
                r1 = await llm_factory_mod.generate_assistance(
                    question="t", product="t", assistance_type="troubleshooting", image=image,
                )
                r2 = await llm_factory_mod.generate_assistance(
                    question="t", product="t", assistance_type="troubleshooting", image=image,
                )
        assert r1 == {"ok": True}
        assert r2 == {"ok": True}
        assert mock_vision.call_count == 2
        # Text-only cache should remain untouched
        assert llm_factory_mod._generate_fn is None

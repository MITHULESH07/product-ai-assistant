import logging
import os

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path=_dotenv_path)
if _dotenv_path:
    logger.info("Loaded environment from %s", _dotenv_path)
else:
    logger.warning("No .env file found — using system environment only")


class Settings:
    def __init__(self):
        self._ollama_base_url: str | None = None
        self._ollama_model: str | None = None
        self._ollama_embedding_model: str | None = None
        self._ollama_timeout: int = 180
        self._cors_origins: list[str] | None = None
        self._max_pdf_upload_size_mb: int = 20
        self._llm_provider: str = "ollama"
        self._nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
        self._nvidia_model: str = "meta/llama-3.1-8b-instruct"
        self._nvidia_vision_model: str | None = None
        self._ollama_vision_model: str | None = None
        self._max_image_size_mb: int = 5
        self._max_image_width: int = 2048
        self._max_image_height: int = 2048
        self._validated: bool = False

    def validate(self) -> None:
        missing = []

        raw_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        if not raw_url:
            missing.append("OLLAMA_BASE_URL")
        self._ollama_base_url = raw_url.rstrip("/")

        raw_model = os.getenv("OLLAMA_MODEL", "").strip()
        if not raw_model:
            missing.append("OLLAMA_MODEL")
        self._ollama_model = raw_model

        self._ollama_embedding_model = (
            os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b").strip()
        )

        raw_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")
        try:
            self._ollama_timeout = int(raw_timeout)
        except (ValueError, TypeError):
            logger.warning("Invalid OLLAMA_TIMEOUT_SECONDS=%r, using 180", raw_timeout)
            self._ollama_timeout = 180

        self._chroma_persist_directory = (
            os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma").strip()
        )
        self._chroma_collection_name = (
            os.getenv("CHROMA_COLLECTION_NAME", "product_manuals").strip()
        )

        raw_max_upload = os.getenv("MAX_PDF_UPLOAD_SIZE_MB", "20")
        try:
            self._max_pdf_upload_size_mb = int(raw_max_upload)
        except (ValueError, TypeError):
            logger.warning("Invalid MAX_PDF_UPLOAD_SIZE_MB=%r, using 20", raw_max_upload)
            self._max_pdf_upload_size_mb = 20

        self._llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        raw_nvidia_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
        self._nvidia_base_url = raw_nvidia_url.rstrip("/")

        self._nvidia_model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct").strip()

        raw_nvidia_vision = os.getenv("NVIDIA_VISION_MODEL", "").strip()
        self._nvidia_vision_model = raw_nvidia_vision or None

        raw_ollama_vision = os.getenv("OLLAMA_VISION_MODEL", "").strip()
        self._ollama_vision_model = raw_ollama_vision or None

        raw_max_image_size = os.getenv("MAX_IMAGE_SIZE_MB", "5")
        try:
            val = int(raw_max_image_size)
            if val <= 0:
                raise ValueError
            self._max_image_size_mb = val
        except (ValueError, TypeError):
            logger.warning("Invalid MAX_IMAGE_SIZE_MB=%r, using 5", raw_max_image_size)
            self._max_image_size_mb = 5

        raw_max_width = os.getenv("MAX_IMAGE_WIDTH", "2048")
        try:
            w = int(raw_max_width)
            if w <= 0:
                raise ValueError
            self._max_image_width = w
        except (ValueError, TypeError):
            logger.warning("Invalid MAX_IMAGE_WIDTH=%r, using 2048", raw_max_width)
            self._max_image_width = 2048

        raw_max_height = os.getenv("MAX_IMAGE_HEIGHT", "2048")
        try:
            h = int(raw_max_height)
            if h <= 0:
                raise ValueError
            self._max_image_height = h
        except (ValueError, TypeError):
            logger.warning("Invalid MAX_IMAGE_HEIGHT=%r, using 2048", raw_max_height)
            self._max_image_height = 2048

        raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        self._cors_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]

        if self._llm_provider == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
            logger.warning(
                "LLM provider is 'nvidia' but NVIDIA_API_KEY is not set in "
                "the system environment. Generation will fail with a "
                "meaningful error message."
            )

        if missing:
            msg = (
                "Missing required environment variables: "
                f"{', '.join(missing)}. "
                "Create a backend/.env file. See backend/.env.example."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        self._validated = True
        logger.info(
            "Configuration valid — provider=%s llm_model=%s embed_model=%s timeout=%ds",
            self._llm_provider,
            self._ollama_model,
            self._ollama_embedding_model,
            self._ollama_timeout,
        )
        if self._llm_provider == "nvidia":
            logger.info(
                "NVIDIA provider — chat_url=%s model=%s",
                self.nvidia_chat_url,
                self._nvidia_model,
            )

    @property
    def ollama_base_url(self) -> str:
        if self._ollama_base_url is None:
            raise RuntimeError("Settings not validated. Call validate() first.")
        return self._ollama_base_url

    @property
    def ollama_model(self) -> str:
        if self._ollama_model is None:
            raise RuntimeError("Settings not validated. Call validate() first.")
        return self._ollama_model

    @property
    def ollama_embedding_model(self) -> str:
        if self._ollama_embedding_model is None:
            raise RuntimeError("Settings not validated. Call validate() first.")
        return self._ollama_embedding_model

    @property
    def ollama_timeout(self) -> int:
        return self._ollama_timeout

    @property
    def chroma_persist_directory(self) -> str:
        return self._chroma_persist_directory

    @property
    def chroma_collection_name(self) -> str:
        return self._chroma_collection_name

    @property
    def max_pdf_upload_size_mb(self) -> int:
        return self._max_pdf_upload_size_mb

    @property
    def llm_provider(self) -> str:
        return self._llm_provider

    @property
    def nvidia_base_url(self) -> str:
        return self._nvidia_base_url

    @property
    def nvidia_model(self) -> str:
        return self._nvidia_model

    @property
    def nvidia_chat_url(self) -> str:
        return f"{self._nvidia_base_url}/chat/completions"

    @property
    def cors_origins(self) -> list[str]:
        if self._cors_origins is None:
            return ["http://localhost:5173", "http://127.0.0.1:5173"]
        return self._cors_origins

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"

    @property
    def ollama_tags_url(self) -> str:
        return f"{self.ollama_base_url}/api/tags"

    @property
    def ollama_embed_url(self) -> str:
        return f"{self.ollama_base_url}/api/embed"

    @property
    def nvidia_vision_model(self) -> str | None:
        return self._nvidia_vision_model

    @property
    def ollama_vision_model(self) -> str | None:
        return self._ollama_vision_model

    @property
    def max_image_size_mb(self) -> int:
        return self._max_image_size_mb

    @property
    def max_image_size_bytes(self) -> int:
        return self._max_image_size_mb * 1024 * 1024

    @property
    def max_image_width(self) -> int:
        return self._max_image_width

    @property
    def max_image_height(self) -> int:
        return self._max_image_height


settings = Settings()

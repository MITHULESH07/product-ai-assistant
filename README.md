# Product AI Assistant

A RAG and Agentic AI-Based Multimodal Assistance System for Product Operation, Troubleshooting, and Maintenance.

## Main Features

- Product manual question answering
- Switchable LLM provider: Ollama (local/remote) or NVIDIA (cloud)
- RAG-based document retrieval
- Troubleshooting assistance
- Product operation guidance
- Maintenance recommendations
- Image and error screenshot support
- Source citations

## Technology Stack

### Frontend

- React
- Vite
- Axios

### Backend

- FastAPI
- Python

### AI

- Qwen3 8B (Ollama) or NVIDIA NIM models
- Qwen3 Embedding 0.6B
- Ollama / NVIDIA
- ChromaDB

## LLM Provider Switching

Set `LLM_PROVIDER` in `.env` to switch between providers:

| Provider | Value | Requirements |
|----------|-------|-------------|
| Ollama | `ollama` (default) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, running Ollama server |
| NVIDIA | `nvidia` | `NVIDIA_BASE_URL`, `NVIDIA_MODEL`, `NVIDIA_API_KEY` (system env) |

- Embeddings always use Ollama (requires a local/remote embedding model).
- Generation can use either provider — switching only requires a server restart.
- `NVIDIA_API_KEY` must be set as a system environment variable, not in `.env`.

## Project Structure

```text
frontend/     React application
backend/      FastAPI and AI backend
data/         Manuals, uploads and vector database
docs/         Architecture and project documentation
tests/        Backend and evaluation tests
scripts/      Document ingestion scripts
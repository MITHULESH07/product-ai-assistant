# Product AI Assistant

A RAG and Agentic AI-Based Multimodal Assistance System for Product Operation, Troubleshooting, and Maintenance.

## Main Features

- Product manual question answering
- Local LLM using Ollama
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

- Qwen3 8B
- Qwen3 Embedding 0.6B
- Ollama
- ChromaDB

## Project Structure

```text
frontend/     React application
backend/      FastAPI and AI backend
data/         Manuals, uploads and vector database
docs/         Architecture and project documentation
tests/        Backend and evaluation tests
scripts/      Document ingestion scripts
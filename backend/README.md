# Product AI Assistant — Backend

FastAPI backend for the RAG and Agentic AI-Based Multimodal Assistance System.

## Quick Start

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env

# Start the server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path              | Description              |
|--------|-------------------|--------------------------|
| GET    | `/`               | Root status              |
| GET    | `/api/health`     | Health check             |
| GET    | `/api/products`   | List supported products  |
| POST   | `/api/analyse`    | Analyse a problem        |

### POST /api/analyse

**Request body:**

```json
{
  "question": "Why is my servo not moving?",
  "product": "MG90S Servo",
  "assistance_type": "troubleshooting"
}
```

`assistance_type` defaults to `"auto"`. Allowed values: `auto`, `operation`, `troubleshooting`, `maintenance`.

**Response:**

```json
{
  "intent": "troubleshooting",
  "product": "MG90S Servo",
  "summary": "Received your question about the MG90S Servo: Why is my servo not moving?",
  "possible_causes": ["Loose wiring or poor connection", "..."],
  "steps": ["Check all physical connections...", "..."],
  "warning": "",
  "escalation_required": false,
  "sources": [
    {"document": "MG90S Servo User Manual", "page": 12}
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
backend/
├── app/
│   ├── api/           # Route modules
│   │   ├── analyse.py
│   │   ├── health.py
│   │   ├── products.py
│   │   └── router.py
│   ├── core/          # Config, exceptions, logging
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── schemas/       # Pydantic models
│   │   ├── chat.py
│   │   ├── root.py
│   │   ├── health.py
│   │   ├── products.py
│   │   └── error.py
│   ├── services/      # Business logic
│   │   └── analysis_service.py
│   └── main.py        # Application entry point
├── tests/
│   └── backend/
│       └── test_api.py
├── .env.example
├── requirements.txt
└── README.md
```

## Notes

- AI responses are currently **dummy placeholders**.
- Integrate `analysis_service.route_request()` when connecting RAG, agents, or an LLM.
- The `app/agents/` and `app/llm/` directories contain teammate-owned code for future agent and LLM integration.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Product AI Assistant API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Product AI Assistant backend is running"}


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/analyse")
def analyse(data: dict):
    question = data.get("question", "")
    product = data.get("product", "")

    return {
        "intent": "troubleshooting",
        "product": product,
        "summary": f"Received your question: {question}",
        "possible_causes": [
            "Dummy cause for testing"
        ],
        "steps": [
            "React successfully connected to FastAPI"
        ],
        "warning": "",
        "sources": []
    }
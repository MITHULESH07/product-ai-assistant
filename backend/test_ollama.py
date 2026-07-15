from app.llm.ollama_client import generate_response

answer = generate_response(
    "What is FastAPI?"
)

print(answer)
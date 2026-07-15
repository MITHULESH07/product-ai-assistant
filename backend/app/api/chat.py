from fastapi import APIRouter, HTTPException

from app.llm.ollama_client import generate_response
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/api",
    tags=["AI Assistant"],
)


@router.post(
    "/analyse",
    response_model=ChatResponse,
)
def analyse_problem(request: ChatRequest) -> ChatResponse:
    try:
        answer = generate_response(
            question=request.question,
            product=request.product,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama request failed: {exc}",
        ) from exc

    return ChatResponse(
        intent="general",
        product=request.product,
        summary=answer,
        possible_causes=[],
        steps=[],
        warning="",
        escalation_required=False,
    )
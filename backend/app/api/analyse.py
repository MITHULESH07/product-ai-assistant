import logging

from fastapi import APIRouter, Depends

from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.services.analysis_service import analyse_problem

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Assistant"])


@router.post("/api/analyse", response_model=ChatResponse)
def analyse(request: ChatRequest):
    question = request.question.strip()
    product = request.product.strip()

    result = analyse_problem(
        question=question,
        product=product,
        assistance_type=request.assistance_type,
    )

    sources = None
    if result.get("sources"):
        sources = [Citation(**s) for s in result["sources"]]

    return ChatResponse(
        intent=result["intent"],
        product=result["product"],
        summary=result["summary"],
        possible_causes=result.get("possible_causes", []),
        steps=result.get("steps", []),
        warning=result.get("warning", ""),
        escalation_required=result.get("escalation_required", False),
        sources=sources,
    )

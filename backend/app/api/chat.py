from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.router import AgentRouter

router = APIRouter(
    prefix="/api",
    tags=["AI Assistant"]
)

agent_router = AgentRouter()


@router.post(
    "/analyse",
    response_model=ChatResponse
)
def analyse_problem(request: ChatRequest):

    try:

        result = agent_router.route(

            question=request.question,

            product=request.product

        )

        return ChatResponse(

            intent=result.get("intent", "general"),

            product=request.product,

            summary=result.get("summary", ""),

            possible_causes=result.get("possible_causes", []),

            steps=result.get("steps", []),

            warning=result.get("warning", ""),

            escalation_required=result.get(
                "escalation_required",
                False
            )

        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
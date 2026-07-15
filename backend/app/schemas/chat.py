from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    product: str = Field(min_length=1, default="General")
    assistance_type: Literal[
        "auto",
        "operation",
        "troubleshooting",
        "maintenance",
    ] = "auto"


class Citation(BaseModel):
    document: str
    page: int


class ChatResponse(BaseModel):
    intent: str
    product: str
    summary: str
    possible_causes: list[str]
    steps: list[str]
    warning: str
    escalation_required: bool
    sources: list[Citation] | None = None

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    product: str = Field(min_length=1, max_length=100)
    assistance_type: Literal[
        "troubleshooting",
        "operation",
        "maintenance",
    ]


class AnalysisResponse(BaseModel):
    intent: str
    product: str
    summary: str
    possible_causes: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    warning: Optional[str] = None
    escalation_required: bool = False
    sources: list[str] = Field(default_factory=list)

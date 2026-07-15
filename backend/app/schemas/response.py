from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    intent: str
    product: str
    summary: str

    possible_causes: list[str] = Field(default_factory=list)

    steps: list[str] = Field(default_factory=list)

    warning: str | None = None

    escalation_required: bool = False

    sources: list[str] = Field(default_factory=list)
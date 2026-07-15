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

    image_url: Optional[str] = None
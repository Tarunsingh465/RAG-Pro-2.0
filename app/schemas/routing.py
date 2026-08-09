from typing import Literal

from pydantic import BaseModel, Field


RouteType = Literal[
    "greeting",
    "rag",
    "guardrail",
    "out_of_scope",
]


class RouteDecision(BaseModel):

    route: RouteType = Field(
        description="Selected application route."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence of the routing decision."
    )

    reason: str = Field(
        description="Short explanation for the routing decision."
    )
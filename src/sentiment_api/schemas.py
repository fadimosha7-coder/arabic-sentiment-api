from typing import List

from pydantic import BaseModel, Field, field_validator

from .model import clean_text

MAX_TEXTS = 16
MAX_CHARS = 1000


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_TEXTS)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v):
        for t in v:
            if len(t) > MAX_CHARS:
                raise ValueError(f"text longer than {MAX_CHARS} characters")
            if not clean_text(t):
                raise ValueError("text is empty after removing links and mentions")
        return v


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    results: List[Prediction]
    latency_ms: float
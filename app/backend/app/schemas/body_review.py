from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Part = Literal['person', 'head', 'arm', 'hand', 'torso', 'leg', 'foot']


class CandidateBox(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    label: Part
    box: list[float] = Field(min_length=4, max_length=4)
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self):
        x0, y0, x1, y1 = self.box
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in self.box) or not (x0 < x1 and y0 < y1):
            raise ValueError('invalid candidate rectangle')
        return self


class ReviewRequest(BaseModel):
    operation_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    source_path: str = Field(min_length=1, max_length=4096)
    parts: list[Part] = Field(default=['arm', 'hand', 'torso'], min_length=1, max_length=7)


class ReviewSelection(BaseModel):
    operation_id: str = Field(default='', pattern=r'^$|^[0-9a-f]{32}$')
    candidates: list[CandidateBox] = Field(min_length=1, max_length=16)


class ReviewEdits(BaseModel):
    candidates: list[CandidateBox] = Field(max_length=96)
    selected: list[str] = Field(max_length=16)
    excluded: list[str] = Field(max_length=96)

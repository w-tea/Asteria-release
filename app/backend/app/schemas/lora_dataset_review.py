from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class PersonAnnotation(ReviewModel):
    label: str = Field(max_length=100)
    hair_color: str = Field(max_length=300)
    hair_style: str = Field(max_length=300)
    clothing: str = Field(max_length=500)
    pose: str = Field(max_length=300)


class CaptionFinding(ReviewModel):
    claim: str = Field(min_length=1, max_length=500)
    verdict: Literal['match', 'mismatch', 'unknown']
    evidence: str = Field(min_length=1, max_length=800)


class DatasetAnnotation(ReviewModel):
    person_count: int | None = Field(ge=0, le=100)
    persons: list[PersonAnnotation] = Field(max_length=16)
    background: str = Field(max_length=800)
    caption_findings: list[CaptionFinding] = Field(max_length=32)

    @model_validator(mode='after')
    def count_covers_annotations(self):
        if self.person_count is not None and self.person_count < len(self.persons):
            raise ValueError('review_person_count')
        return self


class ReviewSave(ReviewModel):
    fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    previous_id: str | None = Field(default=None, max_length=64)
    annotation: DatasetAnnotation


class ReviewStart(ReviewModel):
    fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')
    previous_id: str | None = Field(default=None, max_length=64)
    provider_id: str = Field(min_length=1, max_length=128)
    language: Literal['ja', 'en'] = 'ja'

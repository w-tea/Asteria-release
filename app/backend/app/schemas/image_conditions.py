from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ConditionVerdict = Literal['match', 'mismatch', 'unknown']


class ConditionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class ImageCriterion(ConditionModel):
    id: str = Field(min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_-]+$')
    text: str = Field(min_length=1, max_length=500)


class ConditionRequest(ConditionModel):
    generation_id: int = Field(ge=1)
    output_path: str = Field(min_length=1, max_length=4096)
    provider_id: str = Field(min_length=1, max_length=128)
    language: Literal['en', 'ja'] = 'ja'
    criteria: list[ImageCriterion] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def unique_criteria(self):
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError('condition_duplicate_ids')
        return self


class ConditionFinding(ConditionModel):
    id: str = Field(min_length=1, max_length=64)
    verdict: ConditionVerdict
    evidence: str = Field(min_length=1, max_length=800)


class ConditionAssessment(ConditionModel):
    items: list[ConditionFinding] = Field(min_length=1, max_length=16)


class ConditionCorrection(ConditionModel):
    revision: int = Field(ge=1)
    criterion_id: str = Field(min_length=1, max_length=64)
    verdict: ConditionVerdict | None
    reason: str = Field(min_length=1, max_length=800)

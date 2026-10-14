from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.generation import Txt2ImgRequest


class PracticalScene(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    clothing: str = Field(min_length=1, max_length=1000)
    background: str = Field(min_length=1, max_length=1000)
    pose: str = Field(min_length=1, max_length=1000)


class PracticalTestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    run_id: str = Field(min_length=1, max_length=100)
    model_index: int = Field(default=0, ge=0, le=100)
    name: str = Field(min_length=1, max_length=120)
    common_prompt: str = Field(min_length=1, max_length=4000)
    learned_features: str = Field(min_length=1, max_length=2000)
    weight: float = Field(default=0.8, ge=-10, le=10)
    seed_count: int = Field(default=2, ge=2, le=4)
    reviewed_training: Literal[True]
    training_fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')
    base: Txt2ImgRequest
    scenes: list[PracticalScene] = Field(min_length=1, max_length=8)

    @model_validator(mode='before')
    @classmethod
    def use_dedicated_prompt_when_generate_is_empty(cls, data):
        if isinstance(data, dict) and isinstance(data.get('base'), dict) and not data['base'].get('prompt', '').strip():
            return {**data, 'base': {**data['base'], 'prompt': data.get('common_prompt', '')}}
        return data

    @model_validator(mode='after')
    def nonzero_weight(self):
        if not self.weight:
            raise ValueError('comparison_no_difference')
        return self


class PracticalScores(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: Literal['pass', 'fail', 'unknown'] = 'unknown'
    clothing: Literal['pass', 'fail', 'unknown'] = 'unknown'
    background: Literal['pass', 'fail', 'unknown'] = 'unknown'
    pose: Literal['pass', 'fail', 'unknown'] = 'unknown'
    quality: Literal['pass', 'fail', 'unknown'] = 'unknown'


class PracticalTestRating(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pair: int = Field(ge=0, le=31)
    left: PracticalScores
    right: PracticalScores
    note: str = Field(default='', max_length=4000)
    revealed: bool = False

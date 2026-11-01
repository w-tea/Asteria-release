from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.generation import Img2ImgRequest, InpaintRequest, Txt2ImgRequest


REQUEST_TYPES = {'txt2img': Txt2ImgRequest, 'img2img': Img2ImgRequest, 'inpaint': InpaintRequest}


class PromptComparisonChange(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['positive', 'negative', 'embedding', 'lora', 'section', 'style']
    before: str = Field(default='', max_length=8000)
    after: str = Field(default='', max_length=8000)
    polarity: Literal['positive', 'negative'] = 'positive'
    index: int = Field(default=0, ge=0, le=255)
    section: Literal['base', 'character', 'detail'] = 'character'
    weight: float = Field(default=1, ge=-10, le=10)


class PromptComparisonRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(default='', max_length=120)
    mode: Literal['txt2img', 'img2img', 'inpaint'] = 'txt2img'
    base: Txt2ImgRequest | Img2ImgRequest | InpaintRequest
    change: PromptComparisonChange
    pairs: int = Field(default=2, ge=1, le=32)
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode='before')
    @classmethod
    def validate_base(cls, data):
        if isinstance(data, dict) and data.get('mode', 'txt2img') in REQUEST_TYPES:
            data = {**data, 'base': REQUEST_TYPES[data.get('mode', 'txt2img')].model_validate(data.get('base', {}))}
        return data

    @model_validator(mode='after')
    def validate_seeds(self):
        if self.seeds is not None and (len(self.seeds) != self.pairs or len(set(self.seeds)) != self.pairs
                                       or any(seed < 0 or seed > 4294967295 for seed in self.seeds)):
            raise ValueError('comparison_seeds_invalid')
        return self


class PromptComparisonRating(BaseModel):
    model_config = ConfigDict(extra='forbid')

    pair: int = Field(ge=0, le=31)
    selected: Literal['left', 'right', 'tie', 'neither'] | None = None
    note: str = Field(default='', max_length=4000)
    revealed: bool = False

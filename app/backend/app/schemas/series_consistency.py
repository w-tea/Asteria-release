from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.image_conditions import ConditionModel, ImageCriterion


class SeriesImage(ConditionModel):
    generation_id: int = Field(ge=1)
    output_path: str = Field(min_length=1, max_length=4096)


class SeriesRequest(ConditionModel):
    reference: SeriesImage
    candidates: list[SeriesImage] = Field(min_length=1, max_length=23)
    criteria: list[ImageCriterion] = Field(min_length=1, max_length=16)
    provider_id: str = Field(min_length=1, max_length=128)
    language: Literal['ja', 'en'] = 'ja'

    @model_validator(mode='after')
    def unique(self):
        paths = [Path(x.output_path).resolve() for x in [self.reference, *self.candidates]]
        if len(set(paths)) != len(paths):
            raise ValueError('series_duplicate_image')
        if len({x.id for x in self.criteria}) != len(self.criteria):
            raise ValueError('condition_duplicate_ids')
        return self

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


ShortText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)]
PromptText = Annotated[str, StringConstraints(strict=True, max_length=16000)]
SourceField = Literal['single', 'base', 'character', 'detail', 'negative']


class WorkshopModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class WorkshopRequest(WorkshopModel):
    provider_id: str = Field(min_length=1, max_length=128)
    profile: str = Field(default='sdxl', min_length=1, max_length=64)
    language: Literal['en', 'ja'] = 'ja'


class SceneComposeRequest(WorkshopRequest):
    scene: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
    materials: list[Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2400)]] = Field(default_factory=list, max_length=64)
    protected_tokens: list[ShortText] = Field(default_factory=list, max_length=32)


class SceneFields(WorkshopModel):
    base: PromptText
    character: PromptText
    detail: PromptText
    negative: PromptText

    @model_validator(mode='after')
    def nonempty_scene(self):
        if not any(getattr(self, name).strip() for name in ('base', 'character', 'detail')):
            raise ValueError('scene is empty')
        return self


class SceneProposal(WorkshopModel):
    fields: SceneFields
    notes: list[ShortText] = Field(default_factory=list, max_length=12)


class SemanticReviewRequest(WorkshopRequest):
    fields: dict[SourceField, PromptText] = Field(min_length=1, max_length=5)

    @model_validator(mode='after')
    def active_fields(self):
        if 'single' in self.fields and any(key in self.fields for key in ('base', 'character', 'detail')):
            raise ValueError('single and sections cannot both be active')
        if not any(text.strip() for text in self.fields.values()):
            raise ValueError('input is empty')
        return self


class EvidenceQuote(WorkshopModel):
    field: SourceField
    text: ShortText


class SemanticFinding(WorkshopModel):
    category: Literal['count', 'appearance', 'clothing', 'action', 'other']
    quotes: list[EvidenceQuote] = Field(min_length=2, max_length=5)
    reason: ShortText
    question: ShortText


class SemanticProposal(WorkshopModel):
    findings: list[SemanticFinding] = Field(max_length=16)

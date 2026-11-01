from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MaterialKind = Literal['person', 'clothing', 'pose', 'background']
MaterialProfile = Literal['any', 'sdxl', 'anima', 'flux1', 'flux2']
Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=120)]
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)]


class MaterialModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class MaterialAsset(MaterialModel):
    kind: Literal['lora', 'embedding']
    path: str = Field(min_length=1, max_length=4096)
    stable_model_id: str | None = Field(default=None, max_length=256)
    weight: float = Field(default=1, ge=-10, le=10, allow_inf_nan=False)
    clip_weight: float = Field(default=1, ge=-10, le=10, allow_inf_nan=False)


class MaterialSource(MaterialModel):
    filename: str = Field(max_length=255)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    provider_id: str = Field(max_length=128)
    model: str = Field(max_length=512)


class PromptMaterialWrite(MaterialModel):
    name: Name
    kind: MaterialKind
    prompt: Text
    profile: MaterialProfile = 'any'
    assets: list[MaterialAsset] = Field(default_factory=list, max_length=16)
    source: MaterialSource | None = None


class PromptMaterialUpdate(PromptMaterialWrite):
    revision: int = Field(ge=1)


class MaterialProposal(MaterialModel):
    name: Name
    prompt: Text
    notes: list[Annotated[str, StringConstraints(max_length=1000)]] = Field(default_factory=list, max_length=8)

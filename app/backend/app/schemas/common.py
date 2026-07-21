from pydantic import BaseModel, Field


class LoraSelection(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=-10, le=10)
    clip_weight: float | None = Field(default=None, ge=-10, le=10)
    block_weight: str = ""
    block_weight_preset: str = ""
    hierarchy_mode: str = "chain"

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ImageEditArtifact(BaseModel):
    path: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ImageEditPixelRect(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ImageEditPercentRect(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0, le=100)
    height: float = Field(gt=0, le=100)


class ImageEditCropOperation(BaseModel):
    type: Literal["crop"] = "crop"
    source: ImageEditArtifact
    output: ImageEditArtifact
    rect_pixels: ImageEditPixelRect
    rect_percent: ImageEditPercentRect


class ImageEditLineage(BaseModel):
    version: Literal[1] = 1
    root: ImageEditArtifact
    current: ImageEditArtifact
    operations: list[ImageEditCropOperation] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_operation_chain(self):
        first = self.operations[0]
        if first.source.path != self.root.path or first.source.sha256 != self.root.sha256:
            raise ValueError("image edit lineage root is disconnected")
        previous = first.output
        for operation in self.operations[1:]:
            if operation.source.path != previous.path or operation.source.sha256 != previous.sha256:
                raise ValueError("image edit lineage operations are disconnected")
            previous = operation.output
        if self.current.path != previous.path or self.current.sha256 != previous.sha256:
            raise ValueError("image edit lineage current artifact is disconnected")
        return self

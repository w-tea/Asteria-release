from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.action_contracts import ActionImageRecord, ActionInput, InlineImage, LibraryRecordId
from app.schemas.body_review import CandidateBox, Part


class LibraryPartSource(ActionInput):
    generation_id: LibraryRecordId
    artifact_id: str = Field(min_length=1, description='An image artifact ID returned by generation.history.get.')


class LibraryPartsCompare(ActionInput):
    images: list[LibraryPartSource] = Field(min_length=1, description='Exact Library images in comparison order.')
    part: Part | None = Field(default=None, description='Optional candidate type; omit to list all saved types.')


class LibraryPartSnapshot(BaseModel):
    review_id: str
    width: int
    height: int
    parts: list[Part]
    truncated: bool = False
    candidates: list[CandidateBox]
    selected_candidate_ids: list[str]


class LibraryPartReviewItem(LibraryPartSource):
    review: LibraryPartSnapshot | None


class LibraryPartsResult(BaseModel):
    items: list[LibraryPartReviewItem]


class LibraryPartCrop(LibraryPartSource):
    review_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    candidate_id: str = Field(min_length=1)


class LibraryPartCropResult(LibraryPartCrop):
    part: Part
    width: int
    height: int
    sha256: str
    image: InlineImage


class LibraryPartDetect(LibraryPartSource):
    parts: list[Part] = Field(default=['arm', 'hand', 'torso'], min_length=1, max_length=7)


class LibraryPartDetectionResult(LibraryPartReviewItem):
    review_id: str
    status: Literal['unconfirmed', 'completed']


class LibraryPartMaskPrepare(LibraryPartSource):
    review_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    candidate_ids: list[str] = Field(min_length=1, max_length=16, description='The exact selected candidate IDs.')

    @field_validator('candidate_ids')
    @classmethod
    def distinct_candidates(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError('Use distinct candidate IDs.')
        return values


class LibraryPartMaskResult(LibraryPartSource):
    status: Literal['unconfirmed', 'completed']
    review_id: str
    init_image_path: str | None = None
    mask: ActionImageRecord | None = None
    reused_existing_mask: bool = False

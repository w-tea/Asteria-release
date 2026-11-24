from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict

from app.schemas.image_check import ImageCheckSettings
from app.services.user_settings import AIProviderSettings, OutputSettings, UserSettings, resolve_output_root


class ImageCheckProviderSnapshot(AIProviderSettings):
    model_config = ConfigDict(frozen=True)


class ImageCheckOutputSnapshot(OutputSettings):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True)
class ImageCheckOperation:
    settings: ImageCheckSettings
    providers: tuple[ImageCheckProviderSnapshot, ...]
    output_root: Path
    output_settings: ImageCheckOutputSnapshot | None = None

    def snapshot(self) -> dict:
        return {
            "settings": self.settings.model_dump(mode="json"),
            "providers": [item.model_dump(mode="json") for item in self.providers],
            "output_root": str(self.output_root),
            "output_settings": self.output_settings.model_dump(mode="json") if self.output_settings else {},
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> ImageCheckOperation:
        return cls(
            settings=ImageCheckSettings.model_validate(snapshot["settings"]),
            providers=tuple(ImageCheckProviderSnapshot.model_validate(item) for item in snapshot["providers"]),
            output_root=Path(snapshot["output_root"]),
            output_settings=ImageCheckOutputSnapshot.model_validate(snapshot["output_settings"]),
        )


def capture_image_check_operation(
    user_settings: UserSettings,
    *,
    data_dir: Path,
    enabled: bool | None = None,
) -> ImageCheckOperation:
    """Freeze the check and selected provider with the existing common output root."""
    values = user_settings.image_check.model_dump()
    if enabled is not None:
        values["enabled"] = enabled
    settings = ImageCheckSettings.model_validate(values)
    providers = ()
    if settings.enabled and settings.method == "llm":
        providers = tuple(
            ImageCheckProviderSnapshot.model_validate(provider.model_dump())
            for provider in user_settings.ai.providers
            if provider.id == settings.provider_id
        )
    return ImageCheckOperation(
        settings=settings,
        providers=providers,
        output_root=resolve_output_root(data_dir, user_settings.output).resolve(),
        output_settings=ImageCheckOutputSnapshot.model_validate(user_settings.output.model_dump()),
    )

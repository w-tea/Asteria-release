from __future__ import annotations

from dataclasses import dataclass
from fastapi import APIRouter

from app.schemas.extensions import (
    ActionContribution,
    ExtensionManifest,
    PanelContribution,
    WorkspaceContribution,
)
from app.services.operation_executor import OperationExecutor


class ExtensionSDKError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExtensionRouterRegistration:
    contribution_id: str
    router: APIRouter


@dataclass(frozen=True, slots=True)
class ExtensionOperationRegistration:
    contribution_id: str
    kind: str
    executor: OperationExecutor


@dataclass(frozen=True, slots=True)
class ExtensionRegistration:
    extension_id: str
    version: str
    routers: tuple[ExtensionRouterRegistration, ...]
    workspaces: tuple[WorkspaceContribution, ...]
    panels: tuple[PanelContribution, ...]
    actions: tuple[ActionContribution, ...]
    operations: tuple[ExtensionOperationRegistration, ...]


class ExtensionSDKContext:
    __slots__ = (
        "_action_by_id",
        "_manifest",
        "_operation_by_id",
        "_panel_by_id",
        "_registered_ids",
        "_registrations",
        "_router_by_id",
        "_workspace_by_id",
    )

    def __init__(self, manifest: ExtensionManifest) -> None:
        self._manifest = manifest
        self._workspace_by_id = {item.id.casefold(): item for item in manifest.contributions.workspaces}
        self._panel_by_id = {item.id.casefold(): item for item in manifest.contributions.panels}
        self._action_by_id = {
            item.id.casefold(): item
            for item in manifest.contributions.actions
            if item.surface not in {"backend.api_router", "backend.operation"}
        }
        self._router_by_id = {
            item.id.casefold(): item
            for item in manifest.contributions.actions
            if item.surface == "backend.api_router"
        }
        self._operation_by_id = {
            item.id.casefold(): item
            for item in manifest.contributions.actions
            if item.surface == "backend.operation"
        }
        self._registered_ids: set[str] = set()
        self._registrations: list[object] = []

    @property
    def extension_id(self) -> str:
        return self._manifest.id

    @property
    def version(self) -> str:
        return self._manifest.version

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self._manifest.capabilities

    def register_router(
        self,
        contribution_id: str | None = None,
        router: APIRouter | None = None,
    ) -> None:
        if router is None:
            raise ExtensionSDKError("router registration requires an APIRouter")
        contribution = self._select_declared(contribution_id, self._router_by_id, "router")
        if not isinstance(router, APIRouter):
            raise ExtensionSDKError("router registration requires an APIRouter")
        self._claim(contribution.id)
        self._registrations.append(ExtensionRouterRegistration(contribution.id, router))

    def register_workspace(self, contribution_id: str) -> None:
        contribution = self._select_declared(contribution_id, self._workspace_by_id, "workspace")
        self._claim(contribution.id)
        self._registrations.append(contribution)

    def register_panel(self, contribution_id: str) -> None:
        contribution = self._select_declared(contribution_id, self._panel_by_id, "panel")
        self._claim(contribution.id)
        self._registrations.append(contribution)

    def register_action(self, contribution_id: str) -> None:
        contribution = self._select_declared(contribution_id, self._action_by_id, "action")
        self._claim(contribution.id)
        self._registrations.append(contribution)

    def register_operation(self, contribution_id: str, executor: OperationExecutor) -> None:
        contribution = self._select_declared(contribution_id, self._operation_by_id, "operation")
        if (
            not isinstance(getattr(executor, "version", None), str)
            or not callable(getattr(executor, "resource_claims", None))
            or not callable(getattr(executor, "execute", None))
        ):
            raise ExtensionSDKError("operation registration requires an OperationExecutor")
        self._claim(contribution.id)
        self._registrations.append(
            ExtensionOperationRegistration(
                contribution_id=contribution.id,
                kind=f"extension.{self.extension_id}.{contribution.id}",
                executor=executor,
            )
        )

    def finalize(self) -> ExtensionRegistration:
        expected_ids = {
            item.id.casefold()
            for item in (
                *self._manifest.contributions.workspaces,
                *self._manifest.contributions.panels,
                *self._manifest.contributions.actions,
            )
        }
        if self._registered_ids != expected_ids:
            raise ExtensionSDKError("extension did not register every declared contribution")
        return ExtensionRegistration(
            extension_id=self.extension_id,
            version=self.version,
            routers=tuple(item for item in self._registrations if isinstance(item, ExtensionRouterRegistration)),
            workspaces=tuple(item for item in self._registrations if isinstance(item, WorkspaceContribution)),
            panels=tuple(item for item in self._registrations if isinstance(item, PanelContribution)),
            actions=tuple(item for item in self._registrations if isinstance(item, ActionContribution)),
            operations=tuple(item for item in self._registrations if isinstance(item, ExtensionOperationRegistration)),
        )

    def _claim(self, contribution_id: str) -> None:
        normalized = contribution_id.casefold()
        if normalized in self._registered_ids:
            raise ExtensionSDKError("extension contribution was registered more than once")
        self._registered_ids.add(normalized)

    @staticmethod
    def _select_declared(contribution_id: str | None, declarations: dict[str, object], label: str):
        if contribution_id is None:
            available = tuple(declarations.values())
            if len(available) != 1:
                raise ExtensionSDKError(f"{label} contribution ID is required")
            return available[0]
        if not isinstance(contribution_id, str):
            raise ExtensionSDKError(f"{label} contribution ID is invalid")
        contribution = declarations.get(contribution_id.casefold())
        if contribution is None:
            raise ExtensionSDKError(f"{label} contribution is not declared by this extension")
        return contribution

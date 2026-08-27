from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
import stat

from app.schemas.extensions import ExtensionManifest
from app.services.extension_assets import resolve_active_manifest
from app.services.extension_repository import ExtensionRepository


_FRAME_NONCE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_CONTRIBUTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_BUNDLE_BYTES = 4 * 1024 * 1024


class StandardExtensionFrameError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StandardExtensionFrame:
    content: str
    content_security_policy: str


def build_standard_extension_frame(
    repository: ExtensionRepository,
    *,
    extension_id: str,
    version: str,
    contribution_id: str,
    frame_nonce: str,
    startup_result: object,
) -> StandardExtensionFrame:
    if not _FRAME_NONCE.fullmatch(frame_nonce) or not _CONTRIBUTION_ID.fullmatch(contribution_id):
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    if bool(getattr(startup_result, "safe_mode", False)):
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    state = repository.read_state(extension_id)
    if (
        not state.enabled
        or state.active_version != version
        or state.recovery.status != "ready"
        or state.pending_action is not None
    ):
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    loaded = next(
        (
            record
            for record in tuple(getattr(startup_result, "records", ()))
            if record.extension_id == extension_id
            and record.status == "loaded"
            and record.version == version
        ),
        None,
    )
    if loaded is None:
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    manifest, package_root = resolve_active_manifest(repository, extension_id, state)
    if manifest.execution_level != "standard" or not _declares_frontend_contribution(
        manifest,
        contribution_id,
    ):
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    bundle_path = package_root / Path(*manifest.entrypoints.frontend.split("/"))
    try:
        resolved_bundle = _contained_regular_file(package_root, bundle_path)
        bundle = resolved_bundle.read_bytes()
    except (OSError, ValueError) as exc:
        raise StandardExtensionFrameError("extension_standard_frame_not_found") from exc
    if not bundle or len(bundle) > _MAX_BUNDLE_BYTES or b"\x00" in bundle:
        raise StandardExtensionFrameError("extension_standard_frame_not_found")
    encoded = b64encode(bundle).decode("ascii")
    title = html.escape(manifest.name, quote=True)
    bootstrap = _bootstrap_script(
        encoded_bundle=encoded,
        frame_nonce=frame_nonce,
        extension_id=manifest.id,
        contribution_id=contribution_id,
    )
    csp = (
        "default-src 'none'; "
        f"script-src 'nonce-{frame_nonce}' data: blob: 'wasm-unsafe-eval'; "
        "style-src 'unsafe-inline'; img-src data: blob:; media-src blob:; "
        "font-src data:; connect-src 'none'; base-uri 'none'; "
        "form-action 'none'; object-src 'none'; frame-ancestors 'self' "
        "http://127.0.0.1:3000 http://localhost:3000 "
        "http://127.0.0.1:3001 http://localhost:3001 "
        "http://127.0.0.1:5173 http://localhost:5173"
    )
    content = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title>"
        "<style>"
        ":root{color-scheme:light;font-family:Inter,'Segoe UI',sans-serif;color:#182234;background:transparent}"
        ":root[data-theme='dark']{color-scheme:dark;color:#e7edf6}"
        "html,body{margin:0;min-width:0;background:transparent}"
        "body{padding:14px}button,input,select,textarea{font:inherit}"
        "#asteria-extension-root{min-width:0}"
        "</style></head><body><main id=\"asteria-extension-root\"></main>"
        f"<script type=\"module\" nonce=\"{frame_nonce}\">{bootstrap}</script>"
        "</body></html>"
    )
    return StandardExtensionFrame(content=content, content_security_policy=csp)


def _contained_regular_file(root: Path, candidate: Path) -> Path:
    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    candidate_absolute.relative_to(root_absolute)
    current = candidate_absolute
    while True:
        details = current.lstat()
        if (
            current.is_symlink()
            or bool(getattr(os.path, "isjunction", lambda _path: False)(current))
            or bool(getattr(details, "st_file_attributes", 0) & 0x400)
        ):
            raise ValueError("extension standard frame path crosses a reparse point")
        if current == root_absolute:
            break
        current = current.parent
    if not stat.S_ISREG(candidate_absolute.lstat().st_mode):
        raise ValueError("extension standard frame bundle is not a regular file")
    return candidate_absolute


def _declares_frontend_contribution(manifest: ExtensionManifest, contribution_id: str) -> bool:
    normalized = contribution_id.casefold()
    return any(
        item.id.casefold() == normalized
        for item in (
            *manifest.contributions.workspaces,
            *manifest.contributions.panels,
            *(
                item
                for item in manifest.contributions.actions
                if item.surface not in {"backend.api_router", "backend.operation"}
            ),
        )
    )


def _bootstrap_script(
    *,
    encoded_bundle: str,
    frame_nonce: str,
    extension_id: str,
    contribution_id: str,
) -> str:
    return f"""
const CHANNEL = 'asteria.extension.standard.v1';
const FRAME_NONCE = {json.dumps(frame_nonce)};
const EXTENSION_ID = {json.dumps(extension_id)};
const CONTRIBUTION_ID = {json.dumps(contribution_id)};
const frameParameters = new URLSearchParams(location.search);
const frameTheme = frameParameters.get('theme');
const frameLocale = frameParameters.get('locale');
if (frameTheme === 'light' || frameTheme === 'dark') document.documentElement.dataset.theme = frameTheme;
if (frameLocale === 'ja' || frameLocale === 'en') document.documentElement.lang = frameLocale;
const pending = new Map();
const bridge = Object.freeze({{
  request(operation, payload = null) {{
    const requestId = crypto.randomUUID();
    window.parent.postMessage({{
      channel: CHANNEL, type: 'request', nonce: FRAME_NONCE, requestId,
      extensionId: EXTENSION_ID, contributionId: CONTRIBUTION_ID, operation, payload,
    }}, '*');
    return new Promise((resolve, reject) => pending.set(requestId, {{ resolve, reject }}));
  }},
}});
window.addEventListener('message', (event) => {{
  const message = event.data;
  if (event.source !== window.parent || !message || message.channel !== CHANNEL
      || message.type !== 'response' || message.nonce !== FRAME_NONCE
      || typeof message.requestId !== 'string') return;
  const request = pending.get(message.requestId);
  if (!request) return;
  pending.delete(message.requestId);
  if (message.ok === true) request.resolve(message.result);
  else request.reject(new Error(message.error?.code || 'extension_standard_request_failed'));
}});
const source = atob('{encoded_bundle}');
const bytes = Uint8Array.from(source, value => value.charCodeAt(0));
const moduleUrl = URL.createObjectURL(new Blob([bytes], {{ type: 'text/javascript' }}));
try {{
  const loaded = await import(moduleUrl);
  const contract = loaded.asteriaStandardExtension;
  if (!contract || contract.contractVersion !== 'asteria.extension.standard.frontend.v1'
      || typeof contract.mount !== 'function') throw new Error('extension_standard_contract_invalid');
  const root = document.getElementById('asteria-extension-root');
  await contract.mount(Object.freeze({{ root, bridge, extensionId: EXTENSION_ID, contributionId: CONTRIBUTION_ID }}));
  window.parent.postMessage({{ channel: CHANNEL, type: 'ready', nonce: FRAME_NONCE,
    extensionId: EXTENSION_ID, contributionId: CONTRIBUTION_ID }}, '*');
}} catch {{
  window.parent.postMessage({{ channel: CHANNEL, type: 'failed', nonce: FRAME_NONCE,
    extensionId: EXTENSION_ID, contributionId: CONTRIBUTION_ID,
    error: {{ code: 'extension_standard_frame_failed' }} }}, '*');
}} finally {{ URL.revokeObjectURL(moduleUrl); }}
"""

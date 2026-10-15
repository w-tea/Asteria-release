"""One ephemeral, read-only Codex image assessment; never a development agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile

from app.adapters.image_check_image import ImageCheckImageError, prepare_image_check_image
from app.adapters.image_check_llm import ImageCheckLLMReceipt, SYSTEM_RUBRIC, USER_QUERY
from app.schemas.image_check import ImageCheckAssessment, ImageCheckSettings
from app.services.image_check import parse_image_check_response


MAX_STDOUT_BYTES = 256 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_RESULT_BYTES = 64 * 1024
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DISABLED_FEATURES = (
    "apps", "auth_elicitation", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "code_mode", "code_mode_host", "code_mode_only", "computer_use", "default_mode_request_user_input",
    "deferred_executor", "enable_fanout", "enable_mcp_apps", "goals", "hooks", "image_generation",
    "in_app_browser", "memories", "multi_agent", "multi_agent_v2", "plugins", "request_permissions_tool",
    "request_rule", "shell_snapshot", "shell_tool", "skill_env_var_dependency_prompt",
    "skill_mcp_dependency_install", "tool_call_mcp_elicitation", "tool_suggest", "unified_exec",
    "workspace_dependencies", "standalone_web_search",
)


@dataclass(frozen=True)
class ImageCheckCodexReceipt(ImageCheckLLMReceipt):
    cli_version: str = ""


@dataclass(frozen=True)
class ImageCheckCodexResult:
    assessment: ImageCheckAssessment | None
    receipt: ImageCheckCodexReceipt


class CodexInvocationFailure(ValueError):
    """Bounded diagnostics stay in memory; no raw CLI response enters history."""
    def __init__(self, stdout: bytes, stderr: bytes):
        message = (stdout + stderr).decode("utf-8", errors="replace").casefold()
        reason = "image_check_codex_model_unavailable" if "model is not supported" in message else "image_check_codex_failed"
        super().__init__(reason)
        self.stdout = stdout
        self.stderr = stderr


def version_supported(text: str) -> bool:
    match = re.fullmatch(r"codex-cli (\d+)\.(\d+)\.(\d+)(?:-[A-Za-z0-9.-]+)?", text.strip())
    return bool(match and tuple(map(int, match.groups())) >= (0, 144, 1))


def _executable_candidates() -> list[Path]:
    candidates: list[Path] = []
    for name in ("codex.exe", "codex", "codex.cmd", "codex.ps1"):
        found = shutil.which(name)
        if found:
            path = Path(found)
            if path.suffix.casefold() not in {".cmd", ".ps1", ".bat", ".js"}:
                candidates.append(path)
            candidates.extend(path.parent.glob("node_modules/@openai/codex/node_modules/@openai/codex-win32-*/vendor/*/bin/codex.exe"))
    if os.name == "nt" and os.environ.get("APPDATA"):
        npm = Path(os.environ["APPDATA"]) / "npm/node_modules/@openai/codex/node_modules/@openai"
        candidates.extend(npm.glob("codex-win32-*/vendor/*/bin/codex.exe"))
    return list(dict.fromkeys(candidates))


def discover_codex() -> tuple[Path | None, str, str]:
    unsupported_version = ""
    for candidate in _executable_candidates():
        try:
            result = subprocess.run([str(candidate), "--version"], capture_output=True,
                timeout=3, creationflags=_NO_WINDOW, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = result.stdout[:1024].decode("utf-8", errors="replace").strip()
        if result.returncode == 0 and version_supported(text):
            return candidate.resolve(), text.removeprefix("codex-cli "), ""
        if result.returncode == 0 and text.startswith("codex-cli "):
            unsupported_version = text.removeprefix("codex-cli ")
    if unsupported_version:
        return None, unsupported_version, "image_check_codex_update_required"
    return None, "", "image_check_codex_unavailable"


def assessment_schema() -> dict:
    schema = ImageCheckAssessment.model_json_schema()
    for item in [schema, *schema.get("$defs", {}).values()]:
        if "properties" in item:
            item["required"] = list(item["properties"])
            for prop in item["properties"].values():
                prop.pop("default", None)
    return schema


def cached_image_models() -> list[str]:
    """Read only public model metadata; never load CLI configuration or credentials."""
    root = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    path = root / "models_cache.json"
    try:
        with path.open("rb") as stream:
            content = stream.read(2 * 1024 * 1024 + 1)
        if len(content) > 2 * 1024 * 1024:
            return []
        body = json.loads(content)
        return sorted({item["slug"] for item in body.get("models", [])
            if isinstance(item, dict) and item.get("visibility") == "list"
            and "image" in item.get("input_modalities", []) and isinstance(item.get("slug"), str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item["slug"])})
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def build_codex_command(executable: Path, settings: ImageCheckSettings, directory: Path) -> list[str]:
    image = str((directory / "image.jpg").resolve())
    profile = '{filesystem={":root"="deny",' + json.dumps(image) + '="read"},network={enabled=false}}'
    config = {
        "approval_policy": '"never"', "project_doc_max_bytes": "0", "web_search": '"disabled"',
        "default_permissions": '"asteria-image-check"', "permissions.asteria-image-check": profile,
        "features": "{" + ",".join(f"{name}=false" for name in _DISABLED_FEATURES) + "}",
        "model_reasoning_effort": '"low"', "mcp_servers": "{}", "notify": "[]",
        "skills.include_instructions": "false", "include_app_instructions": "false",
        "model_providers.openai.request_max_retries": "0", "model_providers.openai.stream_max_retries": "0",
    }
    args = [str(executable), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
        "--strict-config", "--json", "--color", "never", "-C", str(directory), "--model", settings.codex_model,
        "--output-schema", str(directory / "schema.json"), "-o", str(directory / "result.json"), "-i", image]
    for key, value in config.items():
        args.extend(["-c", f"{key}={value}"])
    return [*args, "-"]


def validate_events(raw: bytes) -> None:
    completed = False
    try:
        for line in raw.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get("type", "")
            if kind == "turn.completed":
                completed = True
            elif kind in {"item.started", "item.updated", "item.completed"}:
                if event.get("item", {}).get("type") not in {"agent_message", "reasoning"}:
                    raise ValueError("image_check_codex_action_refused")
            elif kind not in {"thread.started", "turn.started"}:
                raise ValueError("image_check_codex_failed")
    except (json.JSONDecodeError, AttributeError, TypeError):
        raise ValueError("image_check_response_invalid") from None
    if not completed:
        raise ValueError("image_check_response_incomplete")


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> bytes:
    data = bytearray()
    while chunk := await stream.read(4096):
        if len(data) + len(chunk) > limit:
            raise ValueError("image_check_response_too_large")
        data.extend(chunk)
    return bytes(data)


async def _stop_owned_process(process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"
        try:
            await asyncio.to_thread(subprocess.run, [str(killer), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
                creationflags=_NO_WINDOW, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


async def run_codex(args: list[str], *, cwd: Path, prompt: str, timeout: float,
                    cancel_event: asyncio.Event | None) -> bytes:
    if cancel_event is not None and cancel_event.is_set():
        raise ValueError("image_check_cancelled")
    process = await asyncio.create_subprocess_exec(*args, cwd=cwd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        creationflags=_NO_WINDOW, start_new_session=os.name != "nt")
    tasks = []
    try:
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        tasks = [asyncio.create_task(_read_bounded(process.stdout, MAX_STDOUT_BYTES)),
            asyncio.create_task(_read_bounded(process.stderr, MAX_STDERR_BYTES)), asyncio.create_task(process.wait())]
        async with asyncio.timeout(timeout):
            while not all(task.done() for task in tasks):
                if cancel_event is not None and cancel_event.is_set():
                    raise ValueError("image_check_cancelled")
                for task in tasks:
                    if task.done() and task.exception():
                        raise task.exception()
                result_path = cwd / "result.json"
                if result_path.exists() and result_path.stat().st_size > MAX_RESULT_BYTES:
                    raise ValueError("image_check_response_too_large")
                await asyncio.sleep(0.05)
            stdout, stderr, code = await asyncio.gather(*tasks)
        if code:
            raise CodexInvocationFailure(stdout, stderr)
        return stdout
    except TimeoutError:
        raise ValueError("image_check_provider_timeout") from None
    finally:
        await _stop_owned_process(process)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def check_image_with_codex(settings: ImageCheckSettings, source: str | Path, *,
                                 output_root: str | Path, cancel_event: asyncio.Event | None = None,
                                 generated_final: bool = False) -> ImageCheckCodexResult:
    image, version, attempts = None, "", 0

    def result(assessment):
        return ImageCheckCodexResult(assessment, ImageCheckCodexReceipt(provider_id="codex", provider_mode="codex",
            model=settings.codex_model, timeout_seconds=settings.codex_timeout_seconds, request_attempts=attempts,
            image=image.receipt if image else None, cli_version=version))

    if not settings.enabled:
        return result(None)
    try:
        if settings.method != "codex":
            raise ValueError("image_check_method_not_supported")
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("image_check_cancelled")
        executable, version, reason = await asyncio.to_thread(discover_codex)
        if executable is None:
            raise ValueError(reason)
        prepare_options = {"generated_final": True} if generated_final else {}
        image = await asyncio.to_thread(prepare_image_check_image, source, output_root=output_root, **prepare_options)
        parent = Path(image.receipt.output_root) / ".image-check"
        parent.mkdir(exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError("image_check_stage_root_mismatch")
        with tempfile.TemporaryDirectory(prefix="codex-", dir=parent) as path:
            directory = Path(path)
            (directory / "image.jpg").write_bytes(image.content)
            (directory / "schema.json").write_text(json.dumps(assessment_schema()), encoding="utf-8")
            attempts = 1
            events = await run_codex(build_codex_command(executable, settings, directory), cwd=directory,
                prompt=SYSTEM_RUBRIC + "\n" + USER_QUERY, timeout=settings.codex_timeout_seconds,
                cancel_event=cancel_event)
            validate_events(events)
            with (directory / "result.json").open("rb") as stream:
                content = stream.read(MAX_RESULT_BYTES + 1)
            if len(content) > MAX_RESULT_BYTES:
                raise ValueError("image_check_response_too_large")
            return result(parse_image_check_response(content.decode("utf-8-sig")))
    except ImageCheckImageError as exc:
        reason = exc.reason_code
    except ValueError as exc:
        reason = str(exc) if str(exc).startswith("image_check_") else "image_check_response_invalid"
    except (OSError, UnicodeError):
        reason = "image_check_codex_failed"
    return result(ImageCheckAssessment(verdict="unknown", reason_code=reason))

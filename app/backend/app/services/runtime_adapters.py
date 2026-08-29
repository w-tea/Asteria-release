from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from app.services.generation_packet import ForgeLineageTxt2ImgPacketBuilder


class RuntimeAdapterError(Exception):
    pass


class RuntimeAdapter(Protocol):
    id: str

    def execute_txt2img(self, packet: dict, packet_path: Path, output_dir: Path) -> dict:
        pass


class RemasterRuntime(Protocol):
    def generate_remaster(self, request: dict, output_dir: Path) -> dict:
        pass


class PacketFileRuntimeAdapter:
    id = "asteria_native_packet"

    def execute_txt2img(self, packet: dict, packet_path: Path, output_dir: Path) -> dict:
        return {
            "status": "prepared",
            "output_paths": [str(packet_path)],
            "runtime_adapter": self.id,
            "runtime_job_id": packet["runtime_job_id"],
            "packet_path": str(packet_path),
        }


class ExternalProcessRuntimeAdapter:
    """Runs an external generation engine process against an Asteria packet.

    The command is intentionally generic: Asteria appends the packet path and
    output directory as positional arguments, and expects JSON on stdout.
    """

    id = "external_process"

    def __init__(self, command: str) -> None:
        self.command = command

    def execute_txt2img(self, packet: dict, packet_path: Path, output_dir: Path) -> dict:
        if not self.command.strip():
            raise RuntimeAdapterError("ASTERIA_ENGINE_COMMAND is not configured")
        args = [*shlex.split(self.command), str(packet_path), str(output_dir)]
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                timeout=3600,
            )
        except OSError as exc:
            raise RuntimeAdapterError(f"Failed to start generation engine: {exc}") from exc
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            raise RuntimeAdapterError(f"Generation engine failed: {message}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeAdapterError("Generation engine did not return JSON on stdout") from exc
        output_paths = result.get("output_paths")
        if not isinstance(output_paths, list) or not output_paths:
            raise RuntimeAdapterError("Generation engine result must include output_paths")
        if not all(isinstance(path, str) and path for path in output_paths):
            raise RuntimeAdapterError("Generation engine output_paths must contain string paths")
        if not all(self._output_path_is_file(path) for path in output_paths):
            raise RuntimeAdapterError("Generation engine output_paths must point to existing files")
        return {
            "status": result.get("status", "completed"),
            "output_paths": output_paths,
            "runtime_adapter": self.id,
            "runtime_job_id": packet["runtime_job_id"],
            "packet_path": str(packet_path),
            "engine_info": result.get("info"),
        }

    def _output_path_is_file(self, path: str) -> bool:
        try:
            return Path(path).is_file()
        except OSError as exc:
            raise RuntimeAdapterError(f"Generation engine output file check failed: {exc}") from exc


class AsteriaNativeEngineAdapter(ExternalProcessRuntimeAdapter):
    id = "asteria_native_engine"

    def __init__(self) -> None:
        super().__init__(f'"{sys.executable}" -m app.engine.runner')

    def execute_txt2img(self, packet: dict, packet_path: Path, output_dir: Path) -> dict:
        result = super().execute_txt2img(packet, packet_path, output_dir)
        result["runtime_adapter"] = self.id
        return result


class AsteriaGenerationRuntime:
    def __init__(
        self,
        packet_builder: ForgeLineageTxt2ImgPacketBuilder | None = None,
        adapter: RuntimeAdapter | None = None,
    ) -> None:
        self.packet_builder = packet_builder or ForgeLineageTxt2ImgPacketBuilder()
        self.adapter = adapter or PacketFileRuntimeAdapter()

    def generate_txt2img(self, request: dict, output_dir: Path) -> dict:
        self._reject_spatial_prompt(request)
        request.setdefault("settings", {})["mode"] = "txt2img"
        packet = self.packet_builder.build(request)
        packet_path = self._write_packet(packet, output_dir)
        return self.adapter.execute_txt2img(packet, packet_path, output_dir)

    def generate_img2img(self, request: dict, output_dir: Path) -> dict:
        self._reject_spatial_prompt(request)
        request.setdefault("settings", {})["mode"] = "img2img"
        packet = self.packet_builder.build(request)
        packet_path = self._write_packet(packet, output_dir)
        return self.adapter.execute_txt2img(packet, packet_path, output_dir)

    def generate_inpaint(self, request: dict, output_dir: Path) -> dict:
        self._reject_spatial_prompt(request)
        request.setdefault("settings", {})["mode"] = "inpaint"
        packet = self.packet_builder.build(request)
        packet_path = self._write_packet(packet, output_dir)
        return self.adapter.execute_txt2img(packet, packet_path, output_dir)

    def _reject_spatial_prompt(self, request: dict) -> None:
        if request.get("_spatial_runtime_context") is not None:
            raise RuntimeAdapterError("spatial_prompt_requires_comfy_workflow")

    def _write_packet(self, packet: dict, output_dir: Path) -> Path:
        try:
            if output_dir.exists() and not output_dir.is_dir():
                raise RuntimeAdapterError(f"Generation output path is not a directory: {output_dir}")
            return self.packet_builder.write(packet, output_dir)
        except RuntimeAdapterError:
            raise
        except OSError as exc:
            raise RuntimeAdapterError(f"Generation packet write failed: {exc}") from exc

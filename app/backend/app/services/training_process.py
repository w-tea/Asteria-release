"""Run the trainer in a private console and optionally record its exit."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import NamedTuple, TextIO

import psutil


class TrainingReceipt(NamedTuple):
    pid: int
    started_at: float
    exit_code: int | None
    finished_at: str | None


def read_training_receipt(path: Path) -> TrainingReceipt | None:
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                or info.st_size > 4096):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("attempt") != path.parent.name:
            return None
        pid, started_at, code = value["pid"], value["started_at"], value["exit_code"]
        finished_at = value.get("finished_at")
        if finished_at is not None and (
            not isinstance(finished_at, str) or datetime.fromisoformat(finished_at).utcoffset() is None
        ):
            return None
        if (type(pid) is int and pid > 0 and type(started_at) in (int, float)
                and math.isfinite(started_at) and started_at > 0
                and (code is None or type(code) is int)):
            return TrainingReceipt(pid, started_at, code, finished_at)
    except (OSError, ValueError, KeyError, OverflowError):
        pass
    return None


def owned_training_relay(path: Path, receipt: TrainingReceipt) -> psutil.Process | None:
    try:
        process = psutil.Process(receipt.pid)
        if process.create_time() != receipt.started_at:
            return None
        args = process.cmdline()
        if (len(args) >= 8 and Path(args[4]) == Path(__file__).resolve()
                and args[6] == "--receipt" and Path(args[7]) == path and process.is_running()):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def popen_training_process(
    command: list[str], *, cwd: Path, stdout: TextIO, receipt_path: Path | None = None,
) -> subprocess.Popen:
    if os.name != "nt" and receipt_path is None:
        return subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)

    # Preserve the synchronous missing-command error before starting the relay.
    if not shutil.which(command[0]):
        raise FileNotFoundError(command[0])
    args = [sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()), str(os.getpid())]
    if receipt_path is not None:
        args.extend(["--receipt", str(receipt_path)])
    args.extend(command)
    if os.name != "nt":
        return subprocess.Popen(args, cwd=str(cwd), stdout=stdout, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return subprocess.Popen(
        args,
        cwd=str(cwd), stdout=stdout, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup,
    )


def _configure_private_console(invoking_pid: int) -> None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # Windows venv redirectors share the relay's new console. They must not be
    # mistaken for the invoking backend, whose console must remain untouched.
    owners = (ctypes.c_uint32 * 128)()
    count = kernel.GetConsoleProcessList(owners, len(owners))
    if not 0 < count <= len(owners) or os.getpid() not in owners[:count] or invoking_pid in owners[:count]:
        raise OSError("Asteria trainer requires its own console; shared console was not changed")
    if not kernel.SetConsoleCP(65001) or not kernel.SetConsoleOutputCP(65001):
        raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    if len(sys.argv) < 3:
        raise ValueError("Training relay requires a command")
    command = sys.argv[2:]
    environment = os.environ.copy()
    if os.name == "nt":
        _configure_private_console(int(sys.argv[1]))
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        print("Asteria training console: UTF-8 (private)", flush=True)
    receipt_path = None
    if command[0] == "--receipt":
        # Isolated Python excludes the working directory, which belongs to the trainer.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from app.services.atomic_json import write_json_atomic

        receipt_path = Path(command[1])
        command = command[2:]
        receipt = {"attempt": receipt_path.parent.name, "pid": os.getpid(),
                   "started_at": psutil.Process().create_time(), "exit_code": None}
        write_json_atomic(receipt_path, receipt)
    # Keep the trainer's argument vector, cwd, standard streams and exit code.
    # Cancellation owns this relay and its complete descendant tree.
    code = subprocess.call(command, env=environment)
    if receipt_path is not None:
        write_json_atomic(receipt_path, {
            **receipt, "exit_code": code, "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    return code


if __name__ == "__main__":
    raise SystemExit(main())

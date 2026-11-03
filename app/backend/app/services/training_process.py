"""Keep Windows trainer and native subprocess text in one private UTF-8 console."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TextIO


def popen_training_process(command: list[str], *, cwd: Path, stdout: TextIO) -> subprocess.Popen:
    if os.name != "nt":
        return subprocess.Popen(command, cwd=str(cwd), stdout=stdout, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)

    # Preserve the synchronous missing-command error before starting the relay.
    if not shutil.which(command[0]):
        raise FileNotFoundError(command[0])
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return subprocess.Popen(
        [sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()), str(os.getpid()), *command],
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
    if os.name != "nt" or len(sys.argv) < 3:
        raise ValueError("Windows training relay requires a command")
    _configure_private_console(int(sys.argv[1]))
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print("Asteria training console: UTF-8 (private)", flush=True)
    # Keep the trainer's argument vector, cwd, standard streams and exit code.
    # Cancellation owns this relay and its complete descendant tree.
    return subprocess.call(sys.argv[2:], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

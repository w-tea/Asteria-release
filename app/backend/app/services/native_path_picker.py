from __future__ import annotations

import os
import subprocess
import sys


def pick_native_path(
    *,
    target: str,
    title: str,
    initial_path: str | None,
    extensions: list[str],
) -> tuple[str, str | None]:
    if os.name != "nt" or sys.platform != "win32":
        return "unavailable", None
    script = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
$target = $env:ASTERIA_PICK_TARGET
$title = $env:ASTERIA_PICK_TITLE
$initial = $env:ASTERIA_PICK_INITIAL
if ($target -eq 'directory') {
  $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
  $dialog.Description = $title
  $dialog.ShowNewFolderButton = $true
  if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
    $dialog.SelectedPath = $initial
  }
  if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($dialog.SelectedPath)
  }
  exit 0
}
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = $title
$dialog.CheckFileExists = $true
$dialog.Multiselect = $false
if ($initial) {
  if (Test-Path -LiteralPath $initial -PathType Leaf) {
    $dialog.InitialDirectory = Split-Path -LiteralPath $initial -Parent
    $dialog.FileName = Split-Path -LiteralPath $initial -Leaf
  } elseif (Test-Path -LiteralPath $initial -PathType Container) {
    $dialog.InitialDirectory = $initial
  }
}
$extensions = $env:ASTERIA_PICK_EXTENSIONS
if ($extensions) {
  $patterns = ($extensions -split ',') | ForEach-Object { "*.$($_.TrimStart('.'))" }
  $dialog.Filter = "Supported files|$($patterns -join ';')|All files|*.*"
}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Out.Write($dialog.FileName)
}
"""
    environment = os.environ.copy()
    environment.update(
        {
            "ASTERIA_PICK_TARGET": target,
            "ASTERIA_PICK_TITLE": title,
            "ASTERIA_PICK_INITIAL": initial_path or "",
            "ASTERIA_PICK_EXTENSIONS": ",".join(
                item.strip().lstrip(".") for item in extensions if item.strip()
            ),
        }
    )
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-STA",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=1800,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable", None
    selected = completed.stdout.strip()
    if completed.returncode != 0:
        return "unavailable", None
    return ("selected", selected) if selected else ("cancelled", None)

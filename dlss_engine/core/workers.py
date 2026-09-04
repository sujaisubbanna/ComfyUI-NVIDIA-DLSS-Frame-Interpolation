"""Launch the bundled Windows workers without moving Python into Wine."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def validate_binary(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"DLSS runtime file is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path.name} is a Git LFS pointer. Run git lfs pull in the "
            "custom-node directory to download the runtime binaries."
        )
    if not header.startswith(b"MZ"):
        raise RuntimeError(f"Not a Windows runtime binary: {path}")


def worker_launch(path: Path, *args: str) -> tuple[list[str], dict]:
    validate_binary(path)
    command = [str(path.resolve()), *args]
    if sys.platform == "win32":
        return command, {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
    if sys.platform != "linux":
        raise RuntimeError("DLSS workers support Windows or Linux with Wine.")

    env = os.environ.copy()
    prefix = env.get("WINEPREFIX", "")
    if not prefix or not Path(prefix).is_absolute():
        raise RuntimeError(
            "Linux DLSS requires WINEPREFIX to point to an existing, "
            "configured Wine prefix using an absolute path. See docs/linux.md."
        )
    if not (Path(prefix) / "drive_c/windows/system32").is_dir():
        raise RuntimeError(
            "WINEPREFIX is not initialized. Set up Wine, DXVK, "
            "VKD3D-Proton and DXVK-NVAPI as described in docs/linux.md."
        )
    configured = env.get("DLSS_WINE_PATH")
    wine = shutil.which(configured) if configured else (
        shutil.which("wine64") or shutil.which("wine")
    )
    if not wine:
        raise RuntimeError(
            "Wine was not found. Install Wine or set DLSS_WINE_PATH to "
            "its executable (a path, not a shell command)."
        )
    # Wine diagnostics belong on stderr, never on the binary protocol stream.
    env.setdefault("WINEDEBUG", "-all")
    return [wine, *command], {"env": env, "start_new_session": True}


class WorkerProcess(subprocess.Popen):
    """Own a render's process group, including children of its Wine loader."""

    def __init__(self, path: Path, *args: str, **kwargs) -> None:
        command, options = worker_launch(path, *args)
        self._owns_group = bool(options.get("start_new_session"))
        super().__init__(command, **options, **kwargs)

    def terminate(self) -> None:
        if self._owns_group:
            self._signal_group(signal.SIGTERM)
        else:
            super().terminate()

    def kill(self) -> None:
        if self._owns_group:
            self._signal_group(signal.SIGKILL)
        else:
            super().kill()

    def _signal_group(self, sig: int) -> None:
        # Never use wineserver -k: it also kills unrelated prefix clients.
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            pass

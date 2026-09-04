from __future__ import annotations

import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DLSSG_DIR = ROOT / "bin" / "runtime" / "dlssg"
DLSSG_RUNTIME = DLSSG_DIR / "nvngx_dlssg.dll"
DLSSG_WORKER = DLSSG_DIR / "dlssg-worker.exe"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
JOBS = ROOT / "jobs"


def _executable(env_name: str, command: str) -> Path:
    configured = os.environ.get(env_name)
    found = configured or shutil.which(command)
    return Path(found).resolve() if found else Path(command)


FFMPEG = _executable("DLSS_FFMPEG_PATH", "ffmpeg")
FFPROBE = _executable("DLSS_FFPROBE_PATH", "ffprobe")

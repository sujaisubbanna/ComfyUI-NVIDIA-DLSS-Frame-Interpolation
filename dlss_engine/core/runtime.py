from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_detection import detect_gpus
from .gpu_selection import resolve_runtime_ai_gpu
from .paths import DLSSG_RUNTIME, DLSSG_WORKER, FFMPEG, FFPROBE


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(frame, 3))
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(frame, 2))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(frame, 1))
    return frame


@dataclass(slots=True)
class PreparedRuntime:
    gpu: dict[str, Any]
    gpus: tuple[dict[str, Any], ...]
    runtime_bundle: dict[str, Any]


_PREPARE_LOCK = threading.Lock()
_PREPARED: PreparedRuntime | None = None


def _check_executable(path: Path, label: str) -> None:
    try:
        result = subprocess.run(
            [str(path), "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"{label} was not found. Add it to PATH or set the "
            f"DLSS_{label.upper()}_PATH environment variable."
        ) from exc
    if result.returncode:
        raise RuntimeError(f"{label} could not start: {result.stderr.decode('utf-8', 'replace')[-2000:]}")


def prepare_runtime() -> PreparedRuntime:
    global _PREPARED
    if _PREPARED is not None:
        return _PREPARED
    with _PREPARE_LOCK:
        if _PREPARED is not None:
            return _PREPARED
        for path, label in (
            (DLSSG_WORKER, "DLSSG worker"),
            (DLSSG_RUNTIME, "NVIDIA DLSSG runtime"),
        ):
            if not path.is_file():
                raise RuntimeError(f"{label} is missing: {path}")
        _check_executable(FFMPEG, "FFmpeg")
        _check_executable(FFPROBE, "FFprobe")
        gpus = detect_gpus()
        runtime_bundle = {
            "dlssg": str(DLSSG_RUNTIME.resolve()),
            "worker": str(DLSSG_WORKER.resolve()),
        }
        gpu = resolve_runtime_ai_gpu(gpus, runtime_bundle)
        _PREPARED = PreparedRuntime(gpu=dict(gpu), gpus=tuple(dict(item) for item in gpus), runtime_bundle=runtime_bundle)
        return _PREPARED

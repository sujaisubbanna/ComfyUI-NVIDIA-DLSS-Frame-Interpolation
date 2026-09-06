from __future__ import annotations

from contextlib import suppress
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io

from .dlss_engine.core.composition import (
    compose_sdr, composition_report, validate_detail_strength,
)
from .dlss_engine.core.gpu_selection import resolve_runtime_ai_gpu
from .dlss_engine.core.jobs import active_job, cancel_active_job
from .dlss_engine.core.runtime import (
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    UPSCALING_MODES,
    DLSSFrameSession,
    prepare_runtime,
    resize_fit,
    resolve_native_settings,
    resolve_output_size,
    resolve_upscaling_mode,
    verify_feature_18,
)
from .dlss_engine.frame_interpolation import (
    ENGINE_CHOICES,
    FPS_CHOICES,
    FrameInterpolationOptions,
    interpolate_image_sequence,
)
from .dlss_engine.frame_interpolation.models import resolve_target_rate
from .dlss_engine.video.guides import TemporalGuideGenerator


UPSCALE_FACTORS = {mode["label"]: factor for factor, mode in UPSCALING_MODES.items()}


def _progress_callback():
    progress_bar = comfy.utils.ProgressBar(1000)

    def progress(value: float, _message: str) -> None:
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except comfy.model_management.InterruptProcessingException:
            cancel_active_job()
            raise
        progress_bar.update_absolute(round(float(value) * 1000), 1000)

    return progress_bar, progress


def _neural_options(
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    automatic_mask: bool,
    dlss_model_preset: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        nr_preset=str(nr_preset),
        nr_style=str(nr_style),
        nr_intensity=float(nr_intensity),
        local_tone_strength=float(local_tone_strength),
        local_structure_strength=float(local_structure_strength),
        skin_structure_strength=float(skin_structure_strength),
        automatic_mask=bool(automatic_mask),
        dlss_model_preset=str(dlss_model_preset),
    )


def _neural_inputs() -> list:
    return [
        io.Combo.Input("nr_preset", options=list(NR_PRESETS), default="Default"),
        io.Combo.Input("nr_style", options=list(NR_STYLES), default="Default"),
        io.Float.Input("nr_intensity", default=1.0, min=0.0, max=2.0, step=0.05),
        io.Float.Input("local_tone_strength", default=1.0, min=0.0, max=2.0, step=0.05),
        io.Float.Input("local_structure_strength", default=1.0, min=0.0, max=2.0, step=0.05),
        io.Float.Input("skin_structure_strength", default=-1.0, min=-1.0, max=2.0, step=0.05),
        io.Boolean.Input("automatic_mask", default=False),
        io.Combo.Input("dlss_model_preset", options=list(DLSS_MODEL_PRESETS), default="Default"),
    ]


def _output_detail_strength_input():
    """Composition control shared by IMAGE-only upscale nodes."""
    return io.Float.Input(
        "output_detail_strength", default=1.0, min=1.0, max=2.0,
        step=0.05, optional=True,
        tooltip="SDR output composition: 1 preserves worker output; "
        "2 amplifies brightness changes. Separate from NR intensity.",
    )


def _image_batch_to_rgba(
    image: torch.Tensor, *, minimum_frames: int = 1
) -> tuple[np.ndarray, int]:
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "IMAGE must have shape [batch, height, width, channels] with 1, 3, or 4 channels."
        )
    batch, height, width, channels = map(int, image.shape)
    if batch < minimum_frames or width < 64 or height < 64:
        raise ValueError(
            "DLSS IMAGE sequences require at least "
            f"{minimum_frames} frame(s) with width and height of 64 pixels or more."
        )
    source = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(source).all():
        raise ValueError("IMAGE contains non-finite pixel values.")
    pixels = np.rint(np.clip(source, 0.0, 1.0) * 255.0).astype(np.uint8)
    if channels == 1:
        rgb = np.repeat(pixels, 3, axis=3)
        alpha = np.full((batch, height, width, 1), 255, dtype=np.uint8)
        return np.concatenate((rgb, alpha), axis=3), channels
    if channels == 3:
        alpha = np.full((batch, height, width, 1), 255, dtype=np.uint8)
        return np.concatenate((pixels, alpha), axis=3), channels
    return np.ascontiguousarray(pixels), channels


def _rgba_to_image(rgba: np.ndarray, channels: int) -> torch.Tensor:
    result = rgba if channels == 4 else rgba[..., :3]
    return torch.from_numpy(np.ascontiguousarray(result)).to(torch.float32).div(255.0)


class NvidiaDLSSFrameInterpolation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSImageFrameInterpolation",
            display_name="NVIDIA DLSS Image Frame Interpolation",
            category="image/interpolation",
            description="Interpolates an ordered ComfyUI IMAGE batch with DLSS Frame Generation.",
            inputs=[
                io.Image.Input("images", tooltip="Ordered IMAGE frames, such as VAE Decode output."),
                io.Combo.Input("input_fps", options=list(FPS_CHOICES), default="24"),
                io.Combo.Input("output_fps", options=list(FPS_CHOICES), default="60", tooltip="Fractional choices use exact 1001-based rates."),
                io.Combo.Input("dlss_engine", options=list(ENGINE_CHOICES), default="Auto", tooltip="Auto uses an exact native grid when supported, then cascades when required."),
            ],
            outputs=[
                io.Image.Output("images", display_name="interpolated_images"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def cancel_current_job(cls) -> str:
        return cancel_active_job()

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        input_fps: str,
        output_fps: str,
        dlss_engine: str,
    ) -> io.NodeOutput:
        _progress_bar, progress = _progress_callback()
        rgba, channels = _image_batch_to_rgba(images, minimum_frames=2)
        options = FrameInterpolationOptions(
            target_fps=str(output_fps),
            engine=str(dlss_engine),
        )
        result = interpolate_image_sequence(
            rgba, resolve_target_rate(str(input_fps)), options, progress
        )
        output = _rgba_to_image(result.frames, channels)
        return io.NodeOutput(output, json.dumps(result.report, indent=2))


class NvidiaDLSSImageSequenceUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSImageSequenceUpscale",
            display_name="NVIDIA DLSS Image Sequence Upscale",
            category="image/upscaling",
            description="Upscales an ordered ComfyUI IMAGE batch in memory with DLSS and temporal guides.",
            inputs=[
                io.Image.Input("images", tooltip="Ordered IMAGE frames, such as VAE Decode output."),
                io.Combo.Input("upscale_mode", options=list(UPSCALE_FACTORS), default="1.5× (Quality)"),
                io.Boolean.Input("require_neural_upscaling", default=False, tooltip="Fail instead of returning a larger fallback result when NVIDIA reports neural upscaling inactive."),
                *_neural_inputs(),
                _output_detail_strength_input(),
            ],
            outputs=[
                io.Image.Output("images", display_name="upscaled_images"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        upscale_mode: str,
        require_neural_upscaling: bool,
        nr_preset: str,
        nr_style: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        dlss_model_preset: str,
        output_detail_strength: float = 1.0,
    ) -> io.NodeOutput:
        output_detail_strength = validate_detail_strength(output_detail_strength)
        source_rgba, channels = _image_batch_to_rgba(images)
        batch, input_height, input_width = map(int, source_rgba.shape[:3])

        factor, mode = resolve_upscaling_mode(UPSCALE_FACTORS[str(upscale_mode)])
        output_width, output_height = resolve_output_size(input_width, input_height, factor)
        native = resolve_native_settings(
            _neural_options(
                nr_preset,
                nr_style,
                nr_intensity,
                local_tone_strength,
                local_structure_strength,
                skin_structure_strength,
                automatic_mask,
                dlss_model_preset,
            )
        )
        prepared = prepare_runtime()
        gpu = resolve_runtime_ai_gpu(prepared.gpus, prepared.runtime_bundle, "auto")
        progress_bar = comfy.utils.ProgressBar(batch)
        session: DLSSFrameSession | None = None
        started = time.perf_counter()
        outputs: list[np.ndarray] = []

        with active_job() as controller:
            try:
                session = DLSSFrameSession(
                    input_width=input_width,
                    input_height=input_height,
                    output_width=output_width,
                    output_height=output_height,
                    frame_count=batch,
                    warmup_frames=0,
                    factor=factor,
                    mode=mode,
                    native_settings=native,
                    gpu=gpu,
                    runtime_bundle=prepared.runtime_bundle,
                    controller=controller,
                )
                guides = TemporalGuideGenerator(session.render_width, session.render_height)
                for index, rgba in enumerate(source_rgba):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    prepared_rgba = resize_fit(rgba, session.render_width, session.render_height)
                    guide = guides.process(prepared_rgba)
                    processed, _pts = session.process(
                        index=index,
                        rgba=prepared_rgba,
                        motion=guide.motion,
                        reset=guide.reset,
                        pts=index,
                    )
                    processed = compose_sdr(
                        rgba, processed, output_detail_strength
                    )
                    if channels == 4:
                        processed[..., 3] = cv2.resize(
                            rgba[..., 3],
                            (output_width, output_height),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    outputs.append(processed)
                    progress_bar.update_absolute(index + 1, batch)
                session.close()
                evidence = verify_feature_18(session.worker_logs, session.reshade_log_text())
            except Exception:
                if session is not None and not session.closed:
                    with suppress(Exception):
                        session.abort()
                raise

        elapsed = time.perf_counter() - started
        if require_neural_upscaling and factor > 1.0 and not evidence["nr_upscaling_active"]:
            raise RuntimeError(
                "NVIDIA completed the image processing but reported neural upscaling inactive. "
                "Disable Require Neural Upscaling to accept the native fallback, or change the "
                "source resolution, upscale mode, NVIDIA driver, or runtime configuration."
            )
        report = {
            "status": "success",
            "source_type": "image_sequence",
            "pipeline": "renodx-dlssnr-feature18",
            "feature_id": 18,
            "feature_18_confirmed": True,
            "images_processed": batch,
            "output_composition": composition_report(output_detail_strength),
            "input_dimensions": {"width": input_width, "height": input_height},
            "negotiated_render_dimensions": {"width": session.render_width, "height": session.render_height},
            "output_dimensions": {"width": output_width, "height": output_height},
            "requested_upscaling_factor": factor,
            "dlss_mode": mode["name"],
            "requested_dlss_model_preset": str(dlss_model_preset),
            "applied_dlss_model_preset": session.applied_dlss_model_preset,
            "nr_upscaling_requested": factor > 1.0,
            "nr_upscaling_active": bool(evidence["nr_upscaling_active"]),
            "nr_native_fallback": bool(evidence["nr_native_fallback"]),
            "node_policy": {"require_neural_upscaling": bool(require_neural_upscaling)},
            "carrier_create_result": str(evidence["carrier_create_result"]),
            "native_settings": native,
            "gpu": gpu,
            "elapsed_seconds": elapsed,
            "dlssnr_evidence": evidence["evidence"],
            "worker_log": session.worker_logs,
            "worker_log_dropped_lines": session.worker_log_dropped_lines,
        }
        return io.NodeOutput(
            _rgba_to_image(np.stack(outputs), channels), json.dumps(report, indent=2)
        )


class DLSSVisualEnhancerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NvidiaDLSSFrameInterpolation, NvidiaDLSSImageSequenceUpscale]


async def comfy_entrypoint() -> DLSSVisualEnhancerExtension:
    return DLSSVisualEnhancerExtension()


__all__ = [
    "NvidiaDLSSFrameInterpolation",
    "NvidiaDLSSImageSequenceUpscale",
    "comfy_entrypoint",
]

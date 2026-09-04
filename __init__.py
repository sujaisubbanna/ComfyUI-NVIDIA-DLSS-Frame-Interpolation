from __future__ import annotations

import json
from pathlib import Path
import tempfile

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, Types, io, ui

from .dlss_engine.core.ffmpeg import CODEC_CHOICES, ENCODING_QUALITIES
from .dlss_engine.frame_interpolation import ENGINE_CHOICES, FPS_CHOICES, FrameInterpolationOptions, interpolate_video


CONTAINER_CHOICES = ("MP4", "MKV", "MOV")
RENAME_MODES = ("Auto", "Copy", "Custom")


class NvidiaDLSSFrameInterpolation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NvidiaDLSSFrameInterpolation",
            display_name="NVIDIA DLSS Frame Interpolation",
            category="video/interpolation",
            description="Interpolates a VIDEO with the bundled NVIDIA DLSS Frame Generation runtime. Connect the VIDEO output to ComfyUI's Save Video node.",
            inputs=[
                io.Video.Input("video", tooltip="Video to interpolate."),
                io.Combo.Input("output_fps", options=list(FPS_CHOICES), default="60", tooltip="Fractional choices use exact 1001-based rates."),
                io.Combo.Input("dlss_engine", options=list(ENGINE_CHOICES), default="Auto", tooltip="Auto uses an exact native grid when supported, then cascades when required."),
                io.Combo.Input("encoding_quality", options=list(ENCODING_QUALITIES), default="Max"),
                io.Combo.Input("video_codec", options=list(CODEC_CHOICES), default="H.264", tooltip="Plain codecs use CPU encoding; NVIDIA NVENC choices use the GPU."),
                io.Combo.Input("container", options=list(CONTAINER_CHOICES), default="MP4"),
                io.Combo.Input("rename", options=list(RENAME_MODES), default="Auto", tooltip="Controls the temporary VIDEO filename only. ComfyUI's Save Video node owns the final name."),
                io.String.Input("custom_suffix", default="_DLSSFG", tooltip="Used for the temporary filename only when Rename is Custom."),
                io.Boolean.Input("hdr_mode", default=False, tooltip="10-bit output with input colorspace metadata. Supported by H.265, AV1, and ProRes only."),
            ],
            outputs=[
                io.Video.Output("video", display_name="interpolated_video"),
                io.String.Output("report", display_name="report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: io.Video.Type,
        output_fps: str,
        dlss_engine: str,
        encoding_quality: str,
        video_codec: str,
        container: str,
        rename: str,
        custom_suffix: str,
        hdr_mode: bool,
    ) -> io.NodeOutput:
        progress_bar = comfy.utils.ProgressBar(1000)

        def progress(value: float, _message: str) -> None:
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress_bar.update_absolute(round(float(value) * 1000), 1000)

        temp_base = Path(folder_paths.get_temp_directory()).resolve()
        output_dir = Path(tempfile.mkdtemp(prefix="dlssfg-output-", dir=temp_base))
        report_dir = output_dir / "reports"
        jobs_dir = temp_base / "dlss_frame_interpolation_jobs"
        options = FrameInterpolationOptions(
            target_fps=str(output_fps),
            engine=str(dlss_engine),
            codec=str(video_codec),
            container=str(container),
            quality=str(encoding_quality),
            hdr_mode=bool(hdr_mode),
            rename_mode=str(rename),
            custom_suffix=str(custom_suffix),
        )

        source = video.get_stream_source()
        start_time, duration = video.get_active_trim_window()
        direct_path = isinstance(source, str) and start_time == 0.0 and duration == 0.0
        with tempfile.TemporaryDirectory(prefix="dlssfg-input-", dir=folder_paths.get_temp_directory()) as temp_dir:
            if direct_path:
                input_path = Path(source).resolve()
            else:
                input_path = Path(temp_dir) / "input.mkv"
                video.save_to(str(input_path), format=Types.VideoContainer.MKV, codec=Types.VideoCodec.AUTO)
            result = interpolate_video(
                input_path,
                options,
                progress,
                output_directory=output_dir,
                jobs_directory=jobs_dir,
                logs_directory=report_dir,
            )

        output_path = Path(result.output_path).resolve()
        report_path = Path(result.report_path).resolve()
        report_text = report_path.read_text(encoding="utf-8")
        json.loads(report_text)
        relative = output_path.relative_to(temp_base)
        preview = ui.SavedResult(output_path.name, relative.parent.as_posix(), io.FolderType.temp)
        return io.NodeOutput(
            InputImpl.VideoFromFile(str(output_path)),
            report_text,
            ui=ui.PreviewVideo([preview]),
        )


class DLSSFrameInterpolationExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NvidiaDLSSFrameInterpolation]


async def comfy_entrypoint() -> DLSSFrameInterpolationExtension:
    return DLSSFrameInterpolationExtension()


__all__ = ["NvidiaDLSSFrameInterpolation", "comfy_entrypoint"]

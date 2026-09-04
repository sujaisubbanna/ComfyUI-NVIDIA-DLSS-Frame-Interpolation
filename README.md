# ComfyUI NVIDIA DLSS Frame Interpolation

Native NVIDIA DLSS Frame Generation interpolation for ComfyUI's `VIDEO` type.

The node takes a video, generates intermediate frames with the bundled NVIDIA DLSSG runtime, and returns another connectable `VIDEO`. It does not create a permanent output by itself—connect it to ComfyUI's built-in **Save Video** node when you want to save the result.

```text
Load Video -> NVIDIA DLSS Frame Interpolation -> Save Video
```

## Features

- Native ComfyUI `VIDEO` input and output
- NVIDIA DLSS Frame Generation through the bundled native worker and runtime
- Automatic native-grid or cascaded interpolation
- Exact fractional rates for 23.976, 29.97, and 59.94 FPS
- CPU and NVIDIA NVENC encoding choices
- MP4, MKV, and MOV intermediates
- Optional 10-bit HDR output with compatible codecs
- Original audio, supported subtitles, chapters, and metadata preserved where the selected container allows
- Scene-cut and duplicate-frame handling
- JSON diagnostics as a second node output
- No model downloads, telemetry, analytics, or other network requests

## Requirements

- Windows
- An NVIDIA RTX GPU supported by the included DLSSG runtime
- A current NVIDIA display driver
- ComfyUI with its standard `av`, `numpy`, and OpenCV packages
- FFmpeg and FFprobe available on `PATH`

Hardware-accelerated GPU scheduling (HAGS) is recommended. The node performs a native capability probe before processing and reports the runtime's actual support; GPU branding alone does not guarantee that DLSS Frame Generation will initialize.

No additional Python packages are required for a normal current ComfyUI installation. The DLSSG worker and `nvngx_dlssg.dll` are included inside this repository. FFmpeg and FFprobe are intentionally not redistributed.

## Installation

1. Place or clone this repository into your ComfyUI custom-node directory:

   ```text
   ComfyUI/custom_nodes/ComfyUI-DLSS-Frame-Interpolation
   ```

2. Confirm that both executables are available:

   ```powershell
   ffmpeg -version
   ffprobe -version
   ```

3. Restart ComfyUI.
4. Search for **NVIDIA DLSS Frame Interpolation** under `video/interpolation`.

If FFmpeg is not on `PATH`, define both variables before starting ComfyUI:

```powershell
$env:DLSS_FFMPEG_PATH = "C:\path\to\ffmpeg.exe"
$env:DLSS_FFPROBE_PATH = "C:\path\to\ffprobe.exe"
```

## Usage

1. Add ComfyUI's **Load Video** node.
2. Connect its `VIDEO` output to this node's `video` input.
3. Select the target FPS, DLSS engine, encoding, and container settings.
4. Connect `interpolated_video` to ComfyUI's **Save Video** node.
5. Queue the workflow. The Save Video node controls the permanent output location and filename.

The optional `report_json` output contains the selected interpolation path, capability information, frame counts, scene-cut statistics, encoder details, and elapsed time.

## Node options

| Option | Choices | Behavior |
| --- | --- | --- |
| Output FPS | `23.976`, `25`, `29.97`, `30`, `50`, `59.94`, `60`, `90`, `120` | Uses exact 1001-based rates for fractional choices. The requested output may not exceed 6× the source rate. |
| DLSS engine | `Auto`, `Native DLSSG`, `Cascade` | Auto selects an exact supported native grid and otherwise uses cascaded 2× stages. Native DLSSG requires a supported exact constant-frame-rate ratio. |
| Encoding quality | `Auto (Default)`, `Max`, `Best`, `Good` | Controls the temporary encoded `VIDEO`. Max uses constant-quality encoding; the other modes use calculated target bitrates. |
| Video codec | `H.264`, `H.264 (NVIDIA NVENC)`, `H.265`, `H.265 (NVIDIA NVENC)`, `AV1`, `AV1 (NVIDIA NVENC)`, `ProRes Proxy` | Plain choices use CPU encoding; suffixed choices require NVIDIA NVENC support. |
| Container | `MP4`, `MKV`, `MOV` | Selects the temporary video container. ProRes Proxy requires MOV or MKV. |
| Rename | `Auto`, `Copy`, `Custom` | Retained from the source application and applies only to the temporary intermediate. The Save Video node owns the permanent filename. |
| Custom suffix | Text | Appended to the temporary name when Rename is Custom. |
| HDR Mode | On/off | Enables 10-bit output and carries input color metadata. Available only with H.265, AV1, or ProRes Proxy. |

## Output behavior

The node must encode an intermediate file because ComfyUI's lazy `VIDEO` object needs a streamable source. That intermediate and its JSON report are written only below ComfyUI's temporary directory:

```text
ComfyUI/temp/dlssfg-output-*/
```

Nothing is written to `ComfyUI/output` by this node. ComfyUI may clear temporary files during its normal cleanup or restart process. Use **Save Video** in the same workflow whenever the result must persist.

Only one DLSS interpolation job runs at a time. FFmpeg and native worker processes are launched without visible console windows, and incomplete output is removed when processing fails.

## How engine selection works

- **Auto:** Uses native DLSSG when the source and target rates form an exact multiplier supported by the installed runtime. Otherwise it builds a deterministic cascaded grid and selects the nearest generated timestamp.
- **Native DLSSG:** Forces the runtime's native multi-frame mode. It rejects unsupported or non-exact FPS ratios instead of silently changing the request.
- **Cascade:** Runs one or more 2× DLSSG stages, then selects frames on the exact requested output timeline. This supports targets such as 24 FPS to 60 FPS.

The first and last real frames remain bounded endpoints. Scene cuts reset DLSSG history so frames are not generated across detected cuts.

## Troubleshooting

### Node does not appear

Confirm that this README and `__init__.py` are directly inside:

```text
ComfyUI/custom_nodes/ComfyUI-DLSS-Frame-Interpolation/
```

Then restart ComfyUI and inspect the startup console for an import error.

### FFmpeg or FFprobe was not found

Install FFmpeg and place both executables on `PATH`, or set `DLSS_FFMPEG_PATH` and `DLSS_FFPROBE_PATH` before launching ComfyUI.

### Direct NVIDIA DLSS Frame Generation is unavailable

Update the NVIDIA driver, enable HAGS in Windows graphics settings, restart Windows, and retry. The diagnostic message and `report_json` identify the detected GPU, driver, runtime, signature status, and native multiplier.

### Native DLSSG rejects the requested FPS

Choose **Auto** or **Cascade**. Native mode accepts only an exact constant-frame-rate multiplier supported by the runtime.

### HDR input is rejected

Enable HDR Mode and select H.265, AV1, or ProRes Proxy. H.264 is SDR-only in this node.

### ProRes Proxy with MP4 fails validation

Select MOV or MKV. ProRes Proxy is not supported in MP4.

## Validation

The packaged node has been validated with the portable ComfyUI runtime on Windows:

- Custom-node loading and schema registration passed.
- The bundled DLSSG worker and NVIDIA DLL matched the source package hashes.
- The native capability probe succeeded on an RTX 4060 Ti.
- A real 640×1024, 24 FPS input produced a 60 FPS cascade result with 165 input frames, 413 output frames, and retained AAC audio.
- The generated `VIDEO` was confirmed to live under `ComfyUI/temp`, not `ComfyUI/output`.

This validation demonstrates the tested local configuration; other GPUs, drivers, codecs, resolutions, and HDR sources still depend on their runtime capabilities.

## Credits and licenses

The interpolation implementation is derived from [DLSS 5 Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer) v5.0 by Merserk. Its MIT license is included as [`LICENSE-DLSS-Visual-Enhancer.txt`](LICENSE-DLSS-Visual-Enhancer.txt).

The bundled NVIDIA runtime is governed by the license included at [`bin/runtime/dlssg/LICENSE-NVIDIA-DLSS.txt`](bin/runtime/dlssg/LICENSE-NVIDIA-DLSS.txt). Review that license before redistribution or commercial release. NVIDIA's license includes a commercial-release notification requirement for applications incorporating the DLSS SDK.

FFmpeg and FFprobe are not included in this repository and remain governed by the license of the user's installation.

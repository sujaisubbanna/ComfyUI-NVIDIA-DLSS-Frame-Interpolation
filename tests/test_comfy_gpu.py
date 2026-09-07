"""Opt-in end-to-end tests through the actual ComfyUI IMAGE node entry points."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest


@unittest.skipUnless(
    os.environ.get("DLSS_RUN_COMFY_GPU_TESTS") == "1",
    "Set DLSS_RUN_COMFY_GPU_TESTS=1 and DLSS_COMFYUI_PATH for node GPU tests",
)
class ComfyNodeGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comfy_path = Path(os.environ["DLSS_COMFYUI_PATH"]).resolve()
        if not (comfy_path / "folder_paths.py").is_file():
            raise RuntimeError("DLSS_COMFYUI_PATH must point to ComfyUI")
        sys.path.insert(0, str(comfy_path))
        import torch

        cls.torch = torch
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "dlss_test_nodes", root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        cls.nodes = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.nodes
        spec.loader.exec_module(cls.nodes)

    def image_batch(self, frames=5, height=360, width=640):
        torch = self.torch
        x = torch.linspace(0, 1, width).reshape(1, 1, width, 1)
        y = torch.linspace(0, 1, height).reshape(1, height, 1, 1)
        base = torch.cat((x.expand(1, height, width, 1), y.expand(1, height, width, 1), (1 - x).expand(1, height, width, 1)), dim=3)
        return torch.cat([torch.roll(base, shifts=index * 19, dims=2) for index in range(frames)])

    def test_frame_interpolation_image_batch(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "30", "60", "Auto",
        )
        output = result.result[0]
        self.assertEqual(tuple(output.shape), (10, 360, 640, 3))
        self.assertTrue(self.torch.isfinite(output).all())
        report = json.loads(result.result[2])
        self.assertEqual(report["source_type"], "image_sequence")
        self.assertEqual(report["input_frames"], 5)
        self.assertEqual(report["output_frames"], 10)
        self.assertEqual(result.result[1], 60.0)
        self.assertGreater(report["generated_frames"], 0)

    def test_frame_interpolation_default_input_fps_24(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "24", "60", "Auto",
        )
        report = json.loads(result.result[2])
        self.assertEqual(report["input_fps"], "24")
        self.assertEqual(result.result[1], 60.0)

    def test_frame_interpolation_exact_fractional_output_fps(self):
        result = self.nodes.NvidiaDLSSImageFrameInterpolation.execute(
            self.image_batch(), "24", "59.94", "Auto",
        )
        report = json.loads(result.result[2])
        self.assertEqual(report["output_fps"], "60000/1001")
        self.assertAlmostEqual(result.result[1], 59.94, delta=0.001)

    def test_image_sequence_neural_rendering(self):
        result = self.nodes.NvidiaDLSSImageSequenceUpscale.execute(
            self.image_batch(frames=3), "1.5× (Quality)", False,
            "Default", "Cinematic", 2.0, 1.0, 1.0, -1.0, False, "Default",
            output_detail_strength=2.0,
        )
        output = result.result[0]
        self.assertEqual(tuple(output.shape), (3, 540, 960, 3))
        self.assertTrue(self.torch.isfinite(output).all())
        report = json.loads(result.result[1])
        self.assertEqual(report["source_type"], "image_sequence")
        self.assertTrue(report["feature_18_confirmed"])
        self.assertEqual(report["output_composition"]["output_detail_strength"], 2.0)
        self.assertTrue(report["output_composition"]["applied"])
        self.assertEqual(report["images_processed"], 3)
        self.assertTrue(report["nr_upscaling_active"] or report["nr_native_fallback"])


if __name__ == "__main__":
    unittest.main()

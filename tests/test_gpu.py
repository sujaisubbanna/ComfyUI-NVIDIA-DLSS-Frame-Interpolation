"""Opt-in tests against the real DLSSG worker without video encoding."""

from fractions import Fraction
import os
import unittest

import numpy as np

from dlss_engine.core.jobs import Cancelled, cancel_active_job
from dlss_engine.frame_interpolation import (
    FrameInterpolationOptions,
    interpolate_image_sequence,
)


def image_sequence(frames=5, height=360, width=640):
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, None, :, None]
    y = np.linspace(0, 255, height, dtype=np.uint8)[None, :, None, None]
    base = np.empty((1, height, width, 4), dtype=np.uint8)
    base[..., 0:1] = x
    base[..., 1:2] = y
    base[..., 2:3] = 255 - x
    base[..., 3] = 255
    return np.concatenate([np.roll(base, index * 19, axis=2) for index in range(frames)])


@unittest.skipUnless(
    os.environ.get("DLSS_RUN_GPU_TESTS") == "1",
    "Set DLSS_RUN_GPU_TESTS=1 with an RTX GPU and configured runtime",
)
class GPUInterpolationTests(unittest.TestCase):
    def test_interpolate_image_sequence(self):
        result = interpolate_image_sequence(
            image_sequence(), Fraction(30),
            FrameInterpolationOptions(target_fps="60", engine="Auto"),
        )
        self.assertEqual(result.frames.shape, (10, 360, 640, 4))
        self.assertGreater(result.report["generated_frames"], 0)
        self.assertEqual(result.report["source_type"], "image_sequence")
        self.assertTrue(result.report["capabilities"]["available"])

    def test_image_sequence_can_be_cancelled(self):
        def cancel_after_first_frame(value, _message):
            if value > 0.05:
                cancel_active_job()

        with self.assertRaises(Cancelled):
            interpolate_image_sequence(
                image_sequence(), Fraction(30),
                FrameInterpolationOptions(target_fps="60", engine="Auto"),
                progress=cancel_after_first_frame,
            )


if __name__ == "__main__":
    unittest.main()

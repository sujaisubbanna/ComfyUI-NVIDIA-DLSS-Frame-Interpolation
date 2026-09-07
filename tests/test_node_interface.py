from pathlib import Path
import unittest
from fractions import Fraction

from dlss_engine.frame_interpolation.models import resolve_target_rate


class ImageNodeInterfaceTests(unittest.TestCase):
    def test_registered_node_ids_and_socket_contracts(self):
        source = (Path(__file__).parents[1] / "__init__.py").read_text()

        self.assertIn('node_id="NvidiaDLSSFrameInterpolation"', source)
        self.assertIn('node_id="NvidiaDLSSVideoUpscale"', source)
        self.assertIn('node_id="NvidiaDLSSImageUpscale"', source)
        self.assertIn('node_id="NvidiaDLSSImageFrameInterpolation"', source)
        self.assertIn('node_id="NvidiaDLSSImageSequenceUpscale"', source)

        # Legacy VIDEO nodes remain available.
        self.assertIn('io.Video.Input("video"', source)
        self.assertIn('io.Video.Output("video",', source)
        # Legacy IMAGE node remains available.
        self.assertIn('io.Image.Input("image")', source)
        self.assertIn('io.Image.Output("image",', source)
        # New IMAGE sequence nodes are available.
        self.assertIn('io.Image.Input("images"', source)
        self.assertIn('io.Image.Output("images",', source)

        # New frame interpolation reports selected output fps as a numeric output.
        self.assertIn('display_name="output_fps"', source)
        self.assertIn('io.Float.Output("output_fps",', source)

    def test_default_input_fps_is_supported(self):
        self.assertEqual(resolve_target_rate("24"), Fraction(24, 1))


if __name__ == "__main__":
    unittest.main()

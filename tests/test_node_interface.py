from pathlib import Path
import unittest


class ImageNodeInterfaceTests(unittest.TestCase):
    def test_registered_nodes_do_not_expose_video_sockets(self):
        source = (Path(__file__).parents[1] / "__init__.py").read_text()

        self.assertNotIn("io.Video.", source)
        self.assertIn('node_id="NvidiaDLSSImageFrameInterpolation"', source)
        self.assertIn('node_id="NvidiaDLSSImageSequenceUpscale"', source)
        self.assertIn("io.Image.Input(\"images\"", source)
        self.assertIn("io.Image.Output(\"images\"", source)


if __name__ == "__main__":
    unittest.main()

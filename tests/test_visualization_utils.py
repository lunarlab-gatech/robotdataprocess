import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import unittest
from robotdataprocess.utils import visualization_utils

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestBuildColorPalette(unittest.TestCase):

    def test_palette_length_and_lightness_range(self):
        """ Test that each color gets a 20-entry palette running from dark to light. """
        paletteList = visualization_utils.build_color_palette(['#FF0000', '#00FF00'])
        self.assertEqual(len(paletteList), 2)
        for palette in paletteList:
            self.assertEqual(len(palette), 20)
            # Darkest entry should be near black, lightest near white
            np.testing.assert_allclose(palette[0], (0.0, 0.0, 0.0), atol=1e-6)
            np.testing.assert_allclose(palette[-1], (1.0, 1.0, 1.0), atol=1e-6)

    def test_palette_preserves_hue(self):
        """ Test that the mid-lightness entry roughly matches the input color. """
        paletteList = visualization_utils.build_color_palette(['#FF0000'])
        mid = paletteList[0][9]
        # Pure red at moderate lightness should stay red-dominant
        self.assertGreater(mid[0], mid[1])
        self.assertGreater(mid[0], mid[2])

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestDrawBackgroundImage(unittest.TestCase):

    def test_missing_x_edge_raises(self):
        """ Test ValueError when background_image_x_edge isn't provided. """
        fig, ax = plt.subplots()
        try:
            with self.assertRaises(ValueError):
                visualization_utils.draw_background_image(ax, "unused.png", None, None)
        finally:
            plt.close(fig)

    def test_extent_matches_image_aspect(self):
        """ Test the returned extent respects x_edge and the image's own aspect ratio. """
        img_data = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            mpimg.imsave(f.name, img_data)
            tmp_img = f.name

        fig, ax = plt.subplots()
        try:
            extent = visualization_utils.draw_background_image(ax, tmp_img, 10.0, None)
            x_min, x_max, y_min, y_max = extent
            self.assertAlmostEqual(x_min, -10.0)
            self.assertAlmostEqual(x_max, 10.0)
            # y_extent = x_edge / w * h = 10 / 200 * 100 = 5
            self.assertAlmostEqual(y_min, -5.0)
            self.assertAlmostEqual(y_max, 5.0)
        finally:
            plt.close(fig)
            os.remove(tmp_img)

    def test_extent_offsets_shift_center(self):
        """ Test that background_image_extent_offsets shifts the image center. """
        img_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            mpimg.imsave(f.name, img_data)
            tmp_img = f.name

        fig, ax = plt.subplots()
        try:
            extent = visualization_utils.draw_background_image(ax, tmp_img, 10.0, (100.0, -50.0))
            x_min, x_max, y_min, y_max = extent
            self.assertAlmostEqual(x_min, 90.0)
            self.assertAlmostEqual(x_max, 110.0)
            self.assertAlmostEqual(y_min, -60.0)
            self.assertAlmostEqual(y_max, -40.0)
        finally:
            plt.close(fig)
            os.remove(tmp_img)

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestComputeBoundsAndAspect(unittest.TestCase):

    def test_widens_short_height(self):
        """ Test that a too-tall region gets its width widened to match target_ar. """
        xy = np.array([[0.0, 0.0], [1.0, 10.0]])
        (x_min, x_max), (y_min, y_max) = visualization_utils.compute_bounds_and_aspect([xy], target_ar=1.5)
        width = x_max - x_min
        height = y_max - y_min
        self.assertAlmostEqual(width / height, 1.5, places=5)

    def test_widens_short_width(self):
        """ Test that a too-wide region gets its height widened to match target_ar. """
        xy = np.array([[0.0, 0.0], [10.0, 1.0]])
        (x_min, x_max), (y_min, y_max) = visualization_utils.compute_bounds_and_aspect([xy], target_ar=1.5)
        width = x_max - x_min
        height = y_max - y_min
        self.assertAlmostEqual(width / height, 1.5, places=5)

    def test_combines_multiple_trajectories(self):
        """ Test that bounds are computed across all given trajectories, not just the first. """
        xy1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        xy2 = np.array([[-5.0, -5.0], [5.0, 5.0]])
        (x_min, x_max), (y_min, y_max) = visualization_utils.compute_bounds_and_aspect([xy1, xy2])
        self.assertLessEqual(x_min, -5.0)
        self.assertGreaterEqual(x_max, 5.0)

if __name__ == '__main__':
    unittest.main()

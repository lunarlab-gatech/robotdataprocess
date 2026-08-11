import cv2
import numpy as np
import os
import tempfile
import unittest
import unittest.mock
from robotdataprocess.utils.VideoGenerator import VideoGenerator

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestComputeFrameTimes(unittest.TestCase):

    def setUp(self):
        self.gen = VideoGenerator(np.zeros((10, 10, 3), dtype=np.uint8), fps=10, xlim=(0.0, 1.0), ylim=(0.0, 1.0))

    def test_spans_combined_range(self):
        """ Test that frame times span the earliest start to the latest end across robots. """
        frame_times = self.gen.compute_frame_times([(0.0, 5.0), (2.0, 8.0)], video_duration_sec=2.0)
        self.assertAlmostEqual(frame_times[0], 0.0)
        self.assertAlmostEqual(frame_times[-1], 8.0)
        self.assertEqual(len(frame_times), 20)  # 2.0 sec * 10 fps

    def test_invalid_duration_raises(self):
        """ Test ValueError for non-positive video_duration_sec. """
        with self.assertRaises(ValueError):
            self.gen.compute_frame_times([(0.0, 1.0)], video_duration_sec=0.0)
        with self.assertRaises(ValueError):
            self.gen.compute_frame_times([(0.0, 1.0)], video_duration_sec=-1.0)

    def test_degenerate_range_raises(self):
        """ Test ValueError when the combined timestamp range isn't positive. """
        with self.assertRaises(ValueError):
            self.gen.compute_frame_times([(1.0, 1.0)], video_duration_sec=1.0)

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestInterpolateXYAtTimes(unittest.TestCase):

    def test_linear_interpolation(self):
        """ Test interpolated positions match expected linear interpolation. """
        ts = np.array([0.0, 1.0, 2.0])
        xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        result = VideoGenerator._interpolate_xy_at_times(ts, xy, np.array([0.5, 1.5]))
        np.testing.assert_allclose(result[0], [5.0, 0.0])
        np.testing.assert_allclose(result[1], [10.0, 5.0])

    def test_out_of_range_is_nan(self):
        """ Test that frame times outside the source range become NaN. """
        ts = np.array([1.0, 2.0])
        xy = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = VideoGenerator._interpolate_xy_at_times(ts, xy, np.array([0.0, 1.5, 3.0]))
        self.assertTrue(np.all(np.isnan(result[0])))
        self.assertFalse(np.any(np.isnan(result[1])))
        self.assertTrue(np.all(np.isnan(result[2])))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestWorldToPixel(unittest.TestCase):

    def test_corners_and_center(self):
        """ Test world-to-pixel mapping against known corner/center points. """
        gen = VideoGenerator(np.zeros((100, 200, 3), dtype=np.uint8), fps=10, xlim=(-10.0, 10.0), ylim=(-5.0, 5.0))
        pts = np.array([[-10.0, 5.0], [10.0, -5.0], [0.0, 0.0]])
        px = gen._world_to_pixel(pts)
        np.testing.assert_allclose(px[0], [0.0, 0.0])      # top-left
        np.testing.assert_allclose(px[1], [200.0, 100.0])  # bottom-right
        np.testing.assert_allclose(px[2], [100.0, 50.0])   # center

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestCvtToPixelTuple(unittest.TestCase):

    def test_rounds_to_nearest_int(self):
        """ Test that fractional pixel coordinates round to the nearest int. """
        self.assertEqual(VideoGenerator._cvt_to_pixel_tuple(np.array([2.4, 2.6])), (2, 3))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestDrawConstantTrailSegment(unittest.TestCase):

    def test_draws_nonzero_pixels(self):
        """ Test that a line segment is drawn onto the canvas. """
        canvas = np.zeros((50, 50, 3), dtype=np.uint8)
        VideoGenerator._draw_constant_trail_segment(canvas, np.array([5.0, 25.0]), np.array([45.0, 25.0]),
                                                      (0, 0, 255), width_px=2.0)
        self.assertTrue(np.any(canvas[25, 5:45] > 0))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestDecayFadeAlpha(unittest.TestCase):

    def test_invalid_params_raise(self):
        """ Test ValueError for decay/floor outside their valid ranges. """
        alpha = np.zeros((5, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            VideoGenerator._decay_fade_alpha(alpha, decay=0.0, floor=0.1)
        with self.assertRaises(ValueError):
            VideoGenerator._decay_fade_alpha(alpha, decay=1.0, floor=0.1)
        with self.assertRaises(ValueError):
            VideoGenerator._decay_fade_alpha(alpha, decay=0.9, floor=-0.1)
        with self.assertRaises(ValueError):
            VideoGenerator._decay_fade_alpha(alpha, decay=0.9, floor=1.0)

    def test_untouched_pixels_stay_zero(self):
        """ Test that pixels never drawn on are left at alpha 0. """
        alpha = np.zeros((5, 5), dtype=np.float32)
        VideoGenerator._decay_fade_alpha(alpha, decay=0.5, floor=0.1)
        np.testing.assert_array_equal(alpha, np.zeros((5, 5), dtype=np.float32))

    def test_touched_pixel_decays_to_floor(self):
        """ Test that a touched pixel decays toward floor and then holds there. """
        alpha = np.zeros((3, 3), dtype=np.float32)
        alpha[1, 1] = 1.0
        for _ in range(200):
            VideoGenerator._decay_fade_alpha(alpha, decay=0.9, floor=0.2)
        self.assertAlmostEqual(alpha[1, 1], 0.2, places=3)
        self.assertGreater(alpha[1, 1], 0.0)  # stays touched, doesn't fade to invisible

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestStampFadeTrailSegment(unittest.TestCase):

    def test_sets_alpha_to_one_along_segment(self):
        """ Test that a freshly stamped segment is set to full opacity. """
        alpha = np.zeros((50, 50), dtype=np.float32)
        VideoGenerator._stamp_fade_trail_segment(alpha, np.array([5.0, 25.0]), np.array([45.0, 25.0]), width_px=2.0)
        self.assertTrue(np.any(alpha[25, 5:45] == 1.0))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestBlendAlphaLayer(unittest.TestCase):

    def test_zero_alpha_leaves_frame_unchanged(self):
        """ Test that alpha=0 doesn't change the frame. """
        frame = np.full((5, 5, 3), 100, dtype=np.uint8)
        alpha = np.zeros((5, 5), dtype=np.float32)
        result = VideoGenerator._blend_alpha_layer(frame, alpha, (0, 0, 255))
        np.testing.assert_array_equal(result, frame)

    def test_full_alpha_replaces_with_color(self):
        """ Test that alpha=1 fully replaces pixels with the given color. """
        frame = np.full((5, 5, 3), 100, dtype=np.uint8)
        alpha = np.ones((5, 5), dtype=np.float32)
        result = VideoGenerator._blend_alpha_layer(frame, alpha, (0, 0, 255))
        np.testing.assert_array_equal(result, np.full((5, 5, 3), (0, 0, 255), dtype=np.uint8))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestBlendLightenLayer(unittest.TestCase):

    def test_zero_alpha_leaves_frame_unchanged(self):
        """ Test that alpha=0 doesn't change the frame. """
        frame = np.full((5, 5, 3), 100, dtype=np.uint8)
        alpha = np.zeros((5, 5), dtype=np.float32)
        result = VideoGenerator._blend_lighten_layer(frame, alpha, (0, 0, 255))
        np.testing.assert_array_equal(result, frame)

    def test_full_alpha_replaces_with_lighter_tint(self):
        """ Test that alpha=1 fully replaces pixels with a lighter tint of the given color, not the color itself. """
        frame = np.full((5, 5, 3), 100, dtype=np.uint8)
        alpha = np.ones((5, 5), dtype=np.float32)
        result = VideoGenerator._blend_lighten_layer(frame, alpha, (0, 0, 255), lighten_amount=0.6)
        expected_light_color = (153, 153, 255)  # (0,0,255) blended 60% of the way towards white
        np.testing.assert_array_equal(result, np.full((5, 5, 3), expected_light_color, dtype=np.uint8))
        self.assertFalse(np.array_equal(result, np.full((5, 5, 3), (0, 0, 255), dtype=np.uint8)))

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestDrawGlowDot(unittest.TestCase):

    def test_brightens_around_center(self):
        """ Test that the glow/dot brightens pixels near the given center. """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = VideoGenerator._draw_glow_dot(frame, np.array([50.0, 50.0]), (0, 0, 255),
                                                dot_radius_px=5.0, glow_radius_px=15.0)
        self.assertTrue(np.any(result[50, 50] > 0))

    def test_center_within_margin_still_affects_frame(self):
        """ Test that a center just outside the frame still draws its glow halo inward. """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = VideoGenerator._draw_glow_dot(frame, np.array([-5.0, 50.0]), (0, 0, 255),
                                                dot_radius_px=5.0, glow_radius_px=15.0)
        self.assertTrue(np.any(result[:, :10] > 0))

    def test_glow_luminance_normalized_across_colors(self):
        """ Test that different colors' glow halos reach comparable perceived brightness. """
        center = np.array([50.0, 50.0])
        bright_color_bgr = (0, 255, 255)  # yellow, high raw luminance
        dim_color_bgr = (255, 0, 0)       # blue, low raw luminance

        def sample_glow_luminance(color_bgr):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = VideoGenerator._draw_glow_dot(frame, center, color_bgr, dot_radius_px=5.0, glow_radius_px=15.0)
            px = tuple(int(c) for c in result[50, 59])  # outside the solid dot, inside the glow
            return VideoGenerator._perceived_luminance(px)

        raw_luminance_gap = abs(VideoGenerator._perceived_luminance(bright_color_bgr) -
                                VideoGenerator._perceived_luminance(dim_color_bgr))
        glow_luminance_gap = abs(sample_glow_luminance(bright_color_bgr) - sample_glow_luminance(dim_color_bgr))
        self.assertLess(glow_luminance_gap, raw_luminance_gap / 2)

    def test_dot_color_unaffected_by_glow_scaling(self):
        """ Test that the solid dot itself keeps its original, unscaled color. """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color_bgr = (0, 255, 255)  # yellow, high raw luminance
        result = VideoGenerator._draw_glow_dot(frame, np.array([50.0, 50.0]), color_bgr,
                                                dot_radius_px=5.0, glow_radius_px=15.0)
        np.testing.assert_array_equal(result[50, 50], np.array(color_bgr, dtype=np.uint8))

    def test_center_far_outside_leaves_frame_unchanged(self):
        """ Test that a far off-frame center draws nothing. """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = VideoGenerator._draw_glow_dot(frame, np.array([-1000.0, 50.0]), (0, 0, 255),
                                                dot_radius_px=5.0, glow_radius_px=15.0)
        np.testing.assert_array_equal(result, frame)

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestGenerate(unittest.TestCase):

    def _make_generator(self, save_path=None):
        background = np.full((60, 90, 3), 200, dtype=np.uint8)
        return VideoGenerator(background, fps=5, xlim=(0.0, 9.0), ylim=(0.0, 6.0), save_path=save_path)

    def test_mismatched_lengths_raise(self):
        """ Test ValueError when input lists have different lengths. """
        gen = self._make_generator()
        ts = [np.array([0.0, 1.0])]
        xy = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        with self.assertRaises(ValueError):
            gen.generate(ts, xy, [(255, 0, 0), (0, 255, 0)], video_duration_sec=1.0)
        with self.assertRaises(ValueError):
            gen.generate(ts, xy, [(255, 0, 0)], video_duration_sec=1.0, names=['a', 'b'])

    def test_save_to_file(self):
        """ Test that generate() saves a non-empty video file when save_path is given. """
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            tmp_path = f.name
        os.remove(tmp_path)
        try:
            gen = self._make_generator(save_path=tmp_path)
            ts1 = np.array([0.0, 1.0, 2.0])
            xy1 = np.array([[1.0, 1.0], [5.0, 3.0], [8.0, 5.0]])
            ts2 = np.array([0.5, 1.5])
            xy2 = np.array([[2.0, 4.0], [7.0, 1.0]])
            gen.generate([ts1, ts2], [xy1, xy2], [(255, 0, 0), (0, 255, 0)],
                         video_duration_sec=1.0, names=['R1', 'R2'], title='Test')
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unopenable_writer_raises(self):
        """ Test ValueError when the video writer can't be opened at save_path. """
        gen = self._make_generator(save_path=os.path.join(tempfile.gettempdir(), "nonexistent_dir_xyz", "out.mp4"))
        ts = np.array([0.0, 1.0])
        xy = np.array([[0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            gen.generate([ts], [xy], [(255, 0, 0)], video_duration_sec=1.0)

    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.waitKey', return_value=-1)
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.imshow')
    def test_dot_freezes_instead_of_disappearing_when_robot_finishes_early(self, mock_imshow, mock_waitkey):
        """ Test that a robot's dot stays frozen at its last position after its own trajectory ends, instead of disappearing. """
        gen = self._make_generator(save_path=None)
        ts1 = np.array([0.0, 1.0])  # finishes early, holds still
        xy1 = np.array([[1.0, 1.0], [1.0, 1.0]])
        ts2 = np.array([0.0, 2.0])  # spans the whole video
        xy2 = np.array([[8.0, 5.0], [8.0, 5.0]])
        gen.generate([ts1, ts2], [xy1, xy2], [(255, 0, 0), (0, 255, 0)], video_duration_sec=1.0)  # 5 frames spanning [0, 2]

        last_frame = mock_imshow.call_args_list[-1].args[1]
        px = gen._world_to_pixel(np.array([1.0, 1.0]))
        col, row = int(round(px[0])), int(round(px[1]))
        background_color = self._make_generator().background[row, col]
        self.assertFalse(np.array_equal(last_frame[row, col], background_color))

    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.VideoWriter')
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.destroyAllWindows')
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.waitKey', return_value=-1)  # never pressed ESC
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.imshow')
    def test_live_playback_calls_imshow(self, mock_imshow, mock_waitkey, mock_destroy_all_windows, mock_video_writer):
        """ Test the live-playback branch calls imshow once per frame and cleans up. """
        gen = self._make_generator(save_path=None)
        ts = np.array([0.0, 1.0, 2.0])
        xy = np.array([[1.0, 1.0], [5.0, 3.0], [8.0, 5.0]])
        gen.generate([ts], [xy], [(255, 0, 0)], video_duration_sec=1.0)  # 5 frames at fps=5

        self.assertEqual(mock_imshow.call_count, 5)
        mock_destroy_all_windows.assert_called_once()
        mock_video_writer.assert_not_called()

    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.destroyAllWindows')
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.waitKey', return_value=27)
    @unittest.mock.patch('robotdataprocess.utils.VideoGenerator.cv2.imshow')
    def test_live_playback_breaks_on_escape(self, mock_imshow, mock_waitkey, mock_destroy_all_windows):
        """ Test that pressing ESC (waitKey == 27) stops rendering early. """
        gen = self._make_generator(save_path=None)
        ts = np.array([0.0, 1.0, 2.0])
        xy = np.array([[1.0, 1.0], [5.0, 3.0], [8.0, 5.0]])
        gen.generate([ts], [xy], [(255, 0, 0)], video_duration_sec=1.0)  # 5 frames at fps=5

        self.assertEqual(mock_imshow.call_count, 1)
        mock_destroy_all_windows.assert_called_once()

if __name__ == '__main__':
    unittest.main()

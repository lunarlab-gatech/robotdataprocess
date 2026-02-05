import numpy as np
import os
import unittest
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.Data import ROSMsgLibType


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImageDataEncoding(unittest.TestCase):
    """
    Test the ImageEncoding enum methods in the ImageData base class.
    """

    # ==================== from_str tests ====================

    def test_from_str_mono8(self):
        """ Test from_str with Mono8 encoding. """
        encoding = ImageData.ImageEncoding.from_str("ImageEncoding.Mono8")
        self.assertEqual(encoding, ImageData.ImageEncoding.Mono8)

    def test_from_str_rgb8(self):
        """ Test from_str with RGB8 encoding. """
        encoding = ImageData.ImageEncoding.from_str("ImageEncoding.RGB8")
        self.assertEqual(encoding, ImageData.ImageEncoding.RGB8)

    def test_from_str_32fc1(self):
        """ Test from_str with 32FC1 encoding. """
        encoding = ImageData.ImageEncoding.from_str("ImageEncoding._32FC1")
        self.assertEqual(encoding, ImageData.ImageEncoding._32FC1)

    def test_from_str_invalid(self):
        """ Test from_str raises NotImplementedError for invalid encoding. """
        with self.assertRaises(NotImplementedError):
            ImageData.ImageEncoding.from_str("InvalidEncoding")

    # ==================== from_ros_str tests ====================

    def test_from_ros_str_mono8(self):
        """ Test from_ros_str with mono8 encoding. """
        encoding = ImageData.ImageEncoding.from_ros_str("mono8")
        self.assertEqual(encoding, ImageData.ImageEncoding.Mono8)

    def test_from_ros_str_mono8_uppercase(self):
        """ Test from_ros_str with MONO8 (uppercase) encoding. """
        encoding = ImageData.ImageEncoding.from_ros_str("MONO8")
        self.assertEqual(encoding, ImageData.ImageEncoding.Mono8)

    def test_from_ros_str_rgb8(self):
        """ Test from_ros_str with rgb8 encoding. """
        encoding = ImageData.ImageEncoding.from_ros_str("rgb8")
        self.assertEqual(encoding, ImageData.ImageEncoding.RGB8)

    def test_from_ros_str_32fc1(self):
        """ Test from_ros_str with 32fc1 encoding. """
        encoding = ImageData.ImageEncoding.from_ros_str("32fc1")
        self.assertEqual(encoding, ImageData.ImageEncoding._32FC1)

    # Note: from_ros_str invalid case is already tested in test_ImageDataInMemory.py

    # ==================== from_dtype_and_channels tests ====================

    def test_from_dtype_and_channels_mono8(self):
        """ Test from_dtype_and_channels with uint8 and 1 channel (Mono8). """
        encoding = ImageData.ImageEncoding.from_dtype_and_channels(np.uint8, 1)
        self.assertEqual(encoding, ImageData.ImageEncoding.Mono8)

    def test_from_dtype_and_channels_rgb8(self):
        """ Test from_dtype_and_channels with uint8 and 3 channels (RGB8). """
        encoding = ImageData.ImageEncoding.from_dtype_and_channels(np.uint8, 3)
        self.assertEqual(encoding, ImageData.ImageEncoding.RGB8)

    def test_from_dtype_and_channels_32fc1(self):
        """ Test from_dtype_and_channels with float32 and 1 channel (32FC1). """
        encoding = ImageData.ImageEncoding.from_dtype_and_channels(np.float32, 1)
        self.assertEqual(encoding, ImageData.ImageEncoding._32FC1)

    def test_from_dtype_and_channels_invalid(self):
        """ Test from_dtype_and_channels raises NotImplementedError for invalid combination. """
        with self.assertRaises(NotImplementedError):
            ImageData.ImageEncoding.from_dtype_and_channels(np.float64, 4)

    # ==================== from_pillow_str tests ====================

    def test_from_pillow_str_rgb(self):
        """ Test from_pillow_str with RGB encoding. """
        encoding = ImageData.ImageEncoding.from_pillow_str("RGB")
        self.assertEqual(encoding, ImageData.ImageEncoding.RGB8)

    def test_from_pillow_str_mono(self):
        """ Test from_pillow_str with L (grayscale) encoding. """
        encoding = ImageData.ImageEncoding.from_pillow_str("L")
        self.assertEqual(encoding, ImageData.ImageEncoding.Mono8)

    def test_from_pillow_str_invalid(self):
        """ Test from_pillow_str raises NotImplementedError for invalid encoding. """
        with self.assertRaises(NotImplementedError):
            ImageData.ImageEncoding.from_pillow_str("RGBA")

    # ==================== to_ros_str tests ====================

    def test_to_ros_str_mono8(self):
        """ Test to_ros_str with Mono8 encoding. """
        ros_str = ImageData.ImageEncoding.to_ros_str(ImageData.ImageEncoding.Mono8)
        self.assertEqual(ros_str, 'mono8')

    def test_to_ros_str_rgb8(self):
        """ Test to_ros_str with RGB8 encoding. """
        ros_str = ImageData.ImageEncoding.to_ros_str(ImageData.ImageEncoding.RGB8)
        self.assertEqual(ros_str, 'rgb8')

    def test_to_ros_str_32fc1(self):
        """ Test to_ros_str with 32FC1 encoding. """
        ros_str = ImageData.ImageEncoding.to_ros_str(ImageData.ImageEncoding._32FC1)
        self.assertEqual(ros_str, '32FC1')

    def test_to_ros_str_invalid(self):
        """ Test to_ros_str raises NotImplementedError for invalid encoding. """
        with self.assertRaises(NotImplementedError):
            # Pass a non-ImageEncoding value
            ImageData.ImageEncoding.to_ros_str("invalid")

    # ==================== to_dtype_and_channels tests ====================

    def test_to_dtype_and_channels_mono8(self):
        """ Test to_dtype_and_channels with Mono8 encoding. """
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(ImageData.ImageEncoding.Mono8)
        self.assertEqual(dtype, np.uint8)
        self.assertEqual(channels, 1)

    def test_to_dtype_and_channels_rgb8(self):
        """ Test to_dtype_and_channels with RGB8 encoding. """
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(ImageData.ImageEncoding.RGB8)
        self.assertEqual(dtype, np.uint8)
        self.assertEqual(channels, 3)

    def test_to_dtype_and_channels_32fc1(self):
        """ Test to_dtype_and_channels with 32FC1 encoding. """
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(ImageData.ImageEncoding._32FC1)
        self.assertEqual(dtype, np.float32)
        self.assertEqual(channels, 1)

    def test_to_dtype_and_channels_invalid(self):
        """ Test to_dtype_and_channels raises NotImplementedError for invalid encoding. """
        with self.assertRaises(NotImplementedError):
            ImageData.ImageEncoding.to_dtype_and_channels("invalid")


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImageData(unittest.TestCase):
    """
    Test the ImageData base class methods.
    """

    def test_get_ros_msg_type_invalid(self):
        """ Test get_ros_msg_type raises NotImplementedError for invalid lib_type. """
        with self.assertRaises(NotImplementedError):
            ImageData.get_ros_msg_type(ROSMsgLibType.NONE)

    def test_get_ros_msg_out_of_bounds(self):
        """ Test get_ros_msg raises ValueError for out-of-bounds index. """
        # Create a minimal ImageData instance
        timestamps = np.array([0.1, 0.2, 0.3])
        images = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        image_data = ImageData("test_frame", timestamps, 10, 10, ImageData.ImageEncoding.RGB8, images)

        # Test negative index
        with self.assertRaises(ValueError):
            image_data.get_ros_msg(ROSMsgLibType.ROSBAGS, -1)

        # Test index >= len
        with self.assertRaises(ValueError):
            image_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 3)

        with self.assertRaises(ValueError):
            image_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 100)

    def test_get_ros_msg_unsupported_encoding_mono8(self):
        """ Test get_ros_msg raises NotImplementedError for Mono8 encoding (not yet supported). """
        timestamps = np.array([0.1])
        images = np.zeros((1, 10, 10), dtype=np.uint8)
        image_data = ImageData("test_frame", timestamps, 10, 10, ImageData.ImageEncoding.Mono8, images)

        with self.assertRaises(NotImplementedError):
            image_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)


if __name__ == "__main__":
    unittest.main()

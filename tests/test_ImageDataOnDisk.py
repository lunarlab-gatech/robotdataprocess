import numpy as np
import os
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImageDataOnDisk
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImageDataOnDisk(unittest.TestCase):
    
    def test_from_image_files(self):
        """ Assert the functionality matches that of ImageDataInMemory """

        # Load the image data using both classes
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        image_data_mem = ImageDataOnDisk.from_image_files(folder_path, 'optical')
        image_data_disk = ImageDataInMemory.from_image_files(folder_path, 'optical')

        # Assert that their data matches
        np.testing.assert_equal(image_data_mem.frame_id, image_data_disk.frame_id)
        np.testing.assert_array_equal(image_data_mem.timestamps, image_data_disk.timestamps)
        np.testing.assert_equal(image_data_mem.height, image_data_disk.height)
        np.testing.assert_equal(image_data_mem.width, image_data_disk.width)
        np.testing.assert_equal(image_data_mem.encoding, image_data_disk.encoding)
        np.testing.assert_array_equal(image_data_mem.images, image_data_disk.images)

    def test_lazy_image_array_operations(self):
        """ Test LazyImageArray slicing, boolean masking, setitem, shape, dtype, len. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        data = ImageDataOnDisk.from_image_files(folder_path, 'optical')

        # Test len
        original_len = len(data.images)
        self.assertGreater(original_len, 0)

        # Test shape property
        shape = data.images.shape
        self.assertEqual(shape[0], original_len)

        # Test dtype property
        dtype = data.images.dtype
        self.assertIsNotNone(dtype)

        # Test single integer indexing (loads actual image)
        img = data.images[0]
        self.assertIsInstance(img, np.ndarray)

        # Test slicing
        sliced = data.images[0:1]
        self.assertIsInstance(sliced, ImageDataOnDisk.LazyImageArray)
        self.assertEqual(len(sliced), 1)

        # Test boolean masking
        mask = np.array([True] + [False] * (original_len - 1))
        masked = data.images[mask]
        self.assertIsInstance(masked, ImageDataOnDisk.LazyImageArray)
        self.assertEqual(len(masked), 1)

        # Test __setitem__ raises RuntimeError
        with self.assertRaises(RuntimeError):
            data.images[0] = np.zeros((10, 10))

    def test_from_npy_files(self):
        """ Test loading 32FC1 npy files from disk. """
        folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_from_npy_files', '32fc1').absolute()
        data = ImageDataOnDisk.from_npy_files(folder, 'depth_cam')
        self.assertEqual(data.encoding, ImageDataOnDisk.ImageEncoding._32FC1)
        self.assertGreater(data.len(), 0)
        # Verify a single image can be loaded
        img = data.images[0]
        self.assertEqual(img.shape, (data.height, data.width))


if __name__ == "__main__":
    unittest.main()
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
        mem_data = ImageDataInMemory.from_image_files(folder_path, 'optical')

        # Test len
        original_len = len(data.images)
        self.assertGreater(original_len, 0)
        self.assertEqual(original_len, len(mem_data.images))

        # Test shape property
        shape = data.images.shape
        self.assertEqual(shape[0], original_len)

        # Test dtype property
        dtype = data.images.dtype
        self.assertIsNotNone(dtype)

        # Test single integer indexing matches InMemory
        for i in range(original_len):
            np.testing.assert_array_equal(data.images[i], mem_data.images[i])

        # Test slicing returns correct data
        sliced = data.images[0:2]
        self.assertIsInstance(sliced, ImageDataOnDisk.LazyImageArray)
        self.assertEqual(len(sliced), 2)
        for i in range(len(sliced)):
            np.testing.assert_array_equal(sliced[i], mem_data.images[i])

        # Test boolean masking returns correct data
        mask = np.array([True, False, True] + [False] * (original_len - 3)) if original_len >= 3 \
            else np.array([True] + [False] * (original_len - 1))
        masked = data.images[mask]
        self.assertIsInstance(masked, ImageDataOnDisk.LazyImageArray)
        mem_masked = mem_data.images[mask]
        self.assertEqual(len(masked), len(mem_masked))
        for i in range(len(masked)):
            np.testing.assert_array_equal(masked[i], mem_masked[i])

        # Test __setitem__ raises RuntimeError
        with self.assertRaises(RuntimeError):
            data.images[0] = np.zeros((10, 10))

    def test_from_npy_files(self):
        """ Test loading 32FC1 npy files from disk matches ImageDataInMemory. """
        folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_from_npy_files', '32fc1').absolute()
        data = ImageDataOnDisk.from_npy_files(folder, 'depth_cam')
        mem_data = ImageDataInMemory.from_npy_files(folder, 'depth_cam')

        # Verify metadata matches
        self.assertEqual(data.encoding, mem_data.encoding)
        self.assertEqual(data.height, mem_data.height)
        self.assertEqual(data.width, mem_data.width)
        self.assertEqual(data.len(), mem_data.len())
        np.testing.assert_array_equal(data.timestamps, mem_data.timestamps)

        # Verify every image matches InMemory
        for i in range(data.len()):
            np.testing.assert_array_equal(data.images[i], mem_data.images[i])


if __name__ == "__main__":
    unittest.main()
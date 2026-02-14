import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk, LazyImageArray
import unittest
import cv2
from PIL import Image

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImageDataOnDisk(unittest.TestCase):
    
    def test_from_image_files(self):
        """ Assert the functionality matches that of ImageDataInMemory """

        # Load the image data using both classes
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        image_data = ImageDataOnDisk.from_image_files(folder_path, 'optical')
        image_data_mem = ImageDataInMemory.from_image_files(folder_path, 'optical')

        # Assert that their data matches
        self.assertEqual(image_data.frame_id, image_data_mem.frame_id)
        np.testing.assert_array_equal(image_data.timestamps, image_data_mem.timestamps)
        self.assertEqual(image_data.height, image_data_mem.height)
        self.assertEqual(image_data.width, image_data_mem.width)
        self.assertEqual(image_data.encoding, image_data_mem.encoding)
        for i in range(image_data.len()):
            np.testing.assert_array_equal(image_data.images[i], image_data_mem.images[i])

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
        sliced_images = data.images[0:2]
        self.assertIsInstance(sliced_images, LazyImageArray)
        self.assertEqual(len(sliced_images), 2)
        for i in range(len(sliced_images)):
            np.testing.assert_array_equal(sliced_images[i], mem_data.images[i])

        # Test boolean masking returns correct data
        mask = np.array([True, False, True] + [False] * (original_len - 3)) if original_len >= 3 \
            else np.array([True] + [False] * (original_len - 1))
        
        # Apply mask to images and check
        masked_images = data.images[mask]
        mem_masked_images = mem_data.images[mask]
        self.assertIsInstance(masked_images, LazyImageArray)
        self.assertEqual(len(masked_images), len(mem_masked_images))
        for i in range(len(masked_images)):
            np.testing.assert_array_equal(masked_images[i], mem_masked_images[i])

        # Test __setitem__ raises RuntimeError
        with self.assertRaises(RuntimeError):
            data.images[0] = np.zeros((10, 10))

    def test_to_encoding(self):
        """ Test encoding conversion from RGB8 to BGR8. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        
        # Load RGB images
        image_data = ImageDataOnDisk.from_image_files(folder_path, 'optical')
        self.assertEqual(image_data.encoding, ImageData.ImageEncoding.RGB8)

        # Get original first image
        original_image = image_data.images[0]
        self.assertEqual(len(image_data.images.transformations), 0)

        # Convert to BGR8
        image_data.to_encoding(ImageData.ImageEncoding.BGR8)
        self.assertEqual(image_data.encoding, ImageData.ImageEncoding.BGR8)
        self.assertEqual(len(image_data.images.transformations), 1)

        # Verify the image is now BGR
        bgr_image = image_data.images[0]
        rgb_converted_back = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(original_image, rgb_converted_back)

        # Test converting again does nothing
        image_data.to_encoding(ImageData.ImageEncoding.BGR8)
        self.assertEqual(image_data.encoding, ImageData.ImageEncoding.BGR8)
        # Verify no new transformations were added
        self.assertEqual(len(image_data.images.transformations), 1) 
        # Verify the image data is still the same (no re-transformation)
        re_bgr_image = image_data.images[0]
        np.testing.assert_array_equal(bgr_image, re_bgr_image)


        # Test unsupported conversion from BGR8 to Mono8
        with self.assertRaises(NotImplementedError):
            image_data.to_encoding(ImageData.ImageEncoding.Mono8)
        
        # Test unsupported conversion from Mono8 to BGR8
        mono_folder = Path(Path('.'), 'tests', 'temporary_files', 'test_ImageDataOnDisk', 'mono_images').absolute()
        mono_folder.mkdir(parents=True, exist_ok=True)
        # Create a dummy mono image
        mono_image_path = mono_folder / "1.000000000.png"
        img = Image.new('L', (100, 100)) # 'L' mode for monochrome
        img.save(str(mono_image_path))
        
        mono_image_data = ImageDataOnDisk.from_image_files(mono_folder, 'optical')
        self.assertEqual(mono_image_data.encoding, ImageData.ImageEncoding.Mono8)
        with self.assertRaises(NotImplementedError):
            mono_image_data.to_encoding(ImageData.ImageEncoding.BGR8)

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
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

if __name__ == "__main__":
    unittest.main()
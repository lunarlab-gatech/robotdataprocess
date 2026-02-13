from copy import deepcopy
import cv2
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores
import shutil
from test_utils import safe_urlretrieve
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestImageDataInMemory(unittest.TestCase):
    
    def setUp(self):
        """ 
        Setup paths and download files that will be used 
        for all tests.
        """

        # Get paths to test bags & external messages
        self.path_hercules_bag = Path(Path('.'), 'tests', 'test_bags', 'hercules_test_bag_pruned_3_FINAL').absolute()
        self.path_external_msgs_ros2 = Path(Path('.').parent, 'external_msgs_ros2').absolute()
        self.path_external_msgs_ros1 = Path(Path('.').parent, 'external_msgs_ros1').absolute()

        # Get paths to files within the input bag
        path_hercules_bag_db3 = self.path_hercules_bag / Path("hercules_test_bag_pruned_3_FINAL.db3")
        path_hercules_bag_yaml = self.path_hercules_bag / Path("metadata.yaml")

        # Download the test bag (as its too big for GitHub)
        if not os.path.isfile(path_hercules_bag_db3):
            safe_urlretrieve("https://www.dropbox.com/scl/fi/0ydrblh1uai1lhbrrk6c6/hercules_test_bag_pruned_3_FINAL.db3?rlkey=n27tgr0vuxcyrsyafavlh0aw9&st=i5qixbjo&dl=1", path_hercules_bag_db3)
        if not os.path.isfile(path_hercules_bag_yaml):
            safe_urlretrieve("https://www.dropbox.com/scl/fi/2iu1djmhedy1j4qci53a4/metadata.yaml?rlkey=x0kb00pruubxtbaojht4ui5yl&st=9b41wq07&dl=1", path_hercules_bag_yaml)

    def test_from_ros_str(self):
        """ Make sure that an exception is thrown with a non-valid ROS encoding str"""
        with np.testing.assert_raises(NotImplementedError):
            ImageDataInMemory.ImageEncoding.from_ros_str("fake_name")

    def test_from_npy(self):
        """
        Test that create of ImgData from an .npy file results 
        in the same class as loading from the rosbag.
        """

        # Convert the data into .npy (byproduct of loading ImageData from rosbag)
        save_folder = Path(Path('.'), 'tests', 'test_outputs', 'test_from_npy').absolute()
        rosData = ImageDataInMemory.from_ros2_bag(self.path_hercules_bag, '/hercules_node/Husky2/front_center_Scene/image', save_folder)

        # Load the .npy file
        npyData = ImageDataInMemory.from_npy(save_folder)

        # Make sure the two classes are equivalent
        np.testing.assert_equal(rosData.frame_id, npyData.frame_id)
        np.testing.assert_array_equal(rosData.timestamps, npyData.timestamps)
        np.testing.assert_equal(rosData.height, npyData.height)
        np.testing.assert_equal(rosData.width, npyData.width)
        np.testing.assert_equal(rosData.encoding, npyData.encoding)
        np.testing.assert_array_equal(rosData.images, npyData.images)
    
        # Make sure that there is actually data in the test bag that we are preserving
        if rosData.height == 0 or rosData.width == 0 or len(rosData.images) == 0 \
            or len(rosData.timestamps) == 0 or rosData.frame_id == "":
            self.fail("Test data is not representative of real-time operation!")

    def test_from_npy_files(self):
        """ Test that we load the data properly from images in individual .npy files. """

        # === Test with 32FC1 Images ===
        # Load the npy files
        files_folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_from_npy_files', '32fc1').absolute()
        image_data = ImageDataInMemory.from_npy_files(files_folder, 'Husky1/front_center_DepthPlanar')

        # Make sure it matches what is loaded directly using NumPy
        np.testing.assert_array_equal(image_data.images[2], np.load(files_folder / '0.150000.npy', 'r'))
        np.testing.assert_array_equal(image_data.timestamps.astype(np.float128), [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

        # Make sure it matches data from using external viewer: https://perchance.org/npy-file-viewer
        np.testing.assert_equal(image_data.images[2][1][1], 1.0576000e+04)
        np.testing.assert_equal(image_data.frame_id, 'Husky1/front_center_DepthPlanar')
        np.testing.assert_equal(image_data.height, 480)
        np.testing.assert_equal(image_data.width, 752)
        np.testing.assert_equal(image_data.encoding, ImageDataInMemory.ImageEncoding._32FC1)

    def test_from_image_files(self):
        """
        Test that load the data from image files and saving in the 
        rosbag results in the same data as just loading the image directly.
        """

        # === Test with RGB8 Images ===
        # Load the image data with the class and save to a ROS2 bag
        files_folder = Path(Path('.'), 'tests', 'test_outputs', 'test_from_image_files', 'rgb').absolute()
        image_data = ImageDataInMemory.from_image_files(files_folder, 'Husky1/front_center_Scene')
        bag_path = Path(Path('.'), 'tests', 'test_bags', 'test_from_image_files', 'rgb_data_bag').absolute()
        if os.path.isdir(bag_path):
            os.remove(bag_path / 'rgb_data_bag.db3')
            os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)
        Ros2BagWrapper.write_data_to_rosbag(bag_path, [image_data], ['/cam0'], [None], None)

        # Load that data directly from the rosbag
        npy_folder = Path(Path('.'), 'tests', 'test_outputs', 'test_from_image_files', 'rgb', 'npy').absolute()
        image_data_after = ImageDataInMemory.from_ros2_bag(bag_path, '/cam0', npy_folder)

        # Make sure that the data is what we expect
        np.testing.assert_array_almost_equal(np.arange(0.05, 10.05, 0.05), image_data_after.timestamps.astype(np.float128), 14)
        np.testing.assert_equal('Husky1/front_center_Scene', image_data_after.frame_id)
        np.testing.assert_equal(480, image_data_after.height)
        np.testing.assert_equal(752, image_data_after.width)
        np.testing.assert_equal(ImageDataInMemory.ImageEncoding.RGB8, image_data_after.encoding)

        # Manually load a couple images to compare image data
        def manual_cv2_load(path, cvt=True):
            image_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cvt: return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            else: return image_bgr

        images = np.zeros((3, 480, 752, 3), dtype=np.uint8)
        images[0] = manual_cv2_load(files_folder / "0.050000.png")
        images[1] = manual_cv2_load(files_folder / "0.350000.png")
        images[2] = manual_cv2_load(files_folder / "9.750000.png")

        # Make sure image data and timestamps match what we loaded manually via OpenCV
        np.testing.assert_array_equal(images[0], image_data_after.images[0])
        np.testing.assert_array_equal(images[1], image_data_after.images[6])
        np.testing.assert_array_equal(images[2], image_data_after.images[194])

        # === Test with Mono8 Images ===
        path = Path('.')/'tests'/'files'/'test_ImageData'/'test_from_image_files'/'mono8'
        image_data = ImageDataInMemory.from_image_files(path.absolute(), 'callie')

        images = np.zeros((1, 652, 1196), dtype=np.uint8)
        images[0] = manual_cv2_load(path / "0.0.png", False)

        np.testing.assert_array_equal(images, image_data.images)
        np.testing.assert_array_equal([0.0], image_data.timestamps.astype(np.float128))
        np.testing.assert_equal('callie', image_data.frame_id)
        np.testing.assert_equal(652, image_data.height)
        np.testing.assert_equal(1196, image_data.width)
        np.testing.assert_equal(ImageDataInMemory.ImageEncoding.Mono8, image_data.encoding)

        # === Unsupported Formats ===
        # Test that it properly detects an unsupported format
        path = Path('.')/'tests'/'files'/'test_ImageData'/'test_from_image_files'/'rgba'
        with np.testing.assert_raises(NotImplementedError):
            _ = ImageDataInMemory.from_image_files(path.absolute(), 'N/A')

    def test_to_npy(self):
        """ Make sure the data isn't changed after saving to an .npy file. """

        # === Test with RGB8 Images ===
        # Load the images
        files_folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_to_npy', 'images').absolute()
        image_data = ImageDataInMemory.from_image_files(files_folder, 'Husky1/front_center_Scene')

        # Save it to an .npy file
        save_path = Path(Path('.'), 'tests', 'temporary_files', 'test_ImageData', 'test_to_npy', 'npy').absolute()
        save_path.mkdir(parents=True, exist_ok=True)
        image_data.to_npy(save_path)

        # Load it back from the .npy file
        npy_data = ImageDataInMemory.from_npy(save_path)

        # Ensure the data hasn't changed
        np.testing.assert_array_almost_equal(image_data.images, npy_data.images, 16)
        np.testing.assert_array_almost_equal(image_data.timestamps, npy_data.timestamps, 16)
        np.testing.assert_equal(image_data.frame_id, npy_data.frame_id)
        np.testing.assert_equal(image_data.height, npy_data.height)
        np.testing.assert_equal(image_data.width, npy_data.width)
        np.testing.assert_equal(image_data.encoding, npy_data.encoding)

        # === Test with 32FC1 Images ===
        # Load the npy files
        files_folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_from_npy_files', '32fc1').absolute()
        image_data = ImageDataInMemory.from_npy_files(files_folder, 'Husky1/front_center_DepthPlanar')

        # Save to .npy file and reload
        save_path = Path(Path('.'), 'tests', 'temporary_files', 'test_ImageData', 'test_to_npy', 'npy_depth').absolute()
        save_path.mkdir(parents=True, exist_ok=True)
        image_data.to_npy(save_path)
        npy_data = ImageDataInMemory.from_npy(save_path)

        # Ensure the data hasn't changed
        np.testing.assert_array_almost_equal(image_data.images, npy_data.images, 16)
        np.testing.assert_array_almost_equal(image_data.timestamps, npy_data.timestamps, 16)
        np.testing.assert_equal(image_data.frame_id, npy_data.frame_id)
        np.testing.assert_equal(image_data.height, npy_data.height)
        np.testing.assert_equal(image_data.width, npy_data.width)
        np.testing.assert_equal(image_data.encoding, npy_data.encoding)
        
    def test_crop_data(self):
        """ Ensure the correct data is cropped. """

        # Load the images
        files_folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_crop_data', 'images').absolute()
        image_data = ImageDataInMemory.from_image_files(files_folder, 'camera')

        # Ensure cropping works correctly
        image_data_cropped = deepcopy(image_data)
        image_data_cropped.crop_data(Decimal('0.075'), Decimal('0.15'))
        np.testing.assert_array_equal(image_data_cropped.timestamps, image_data.timestamps[1:])
        np.testing.assert_array_equal(image_data_cropped.images, image_data.images[1:])

        image_data_cropped = deepcopy(image_data)
        image_data_cropped.crop_data(Decimal('0.075'), Decimal('0.125'))
        np.testing.assert_array_equal(image_data_cropped.timestamps, image_data.timestamps[1:2])
        np.testing.assert_array_equal(image_data_cropped.images, image_data.images[1:2])

    def test_get_ros_msg(self):
        """ Test that the ROS message is created properly for images with 32FC1 encoding. 
            TODO: Add test for RGB8 encoding as well. """
        
        # Load the images
        files_folder = Path(Path('.'), 'tests', 'files', 'test_ImageData', 'test_get_ros_msg', 'images').absolute()
        image_data = ImageDataInMemory.from_npy_files(files_folder, 'cam0')

        # Write the image data to a ROS message
        bag_path = Path(Path('.'), 'tests', 'temporary_files', 'test_ImageData', 'test_get_ros_msg', 'depth.bag').absolute()
        if bag_path.exists():
            shutil.rmtree(bag_path)
        Ros2BagWrapper.write_data_to_rosbag(
            bag_path,
            [image_data], 
            ['/cam0/depth'], 
            [None], 
            None)

        # Load that data back from the rosbag
        image_data_after = ImageDataInMemory.from_ros2_bag(bag_path, '/cam0/depth', bag_path.parent / 'npy')

        # Ensure that the data is the same
        np.testing.assert_array_equal(image_data.images, image_data_after.images)
        np.testing.assert_array_almost_equal(image_data.timestamps, image_data_after.timestamps, 16)
        np.testing.assert_equal(image_data.frame_id, image_data_after.frame_id)
        np.testing.assert_equal(image_data.height, image_data_after.height)
        np.testing.assert_equal(image_data.width, image_data_after.width)
        np.testing.assert_equal(image_data.encoding, image_data_after.encoding)

    def helper_from_ros2_bag_matches_raw_messages(self, topic, dtype, expected_shape):
        """ Helper to verify that from_ros2_bag extracts images matching raw ROS2 messages. """

        # Setup output folder
        output_folder = Path(Path('.'), 'tests', 'test_outputs', 'test_from_ros2_bag_matches_raw').absolute()

        # Delete output files if they exist
        file_path_imgs = Path(output_folder, 'imgs.npy')
        if os.path.exists(file_path_imgs):
            os.remove(file_path_imgs)
        file_path_times = Path(output_folder, 'times.npy')
        if os.path.exists(file_path_times):
            os.remove(file_path_times)

        # Extract images using from_ros2_bag
        ImageDataInMemory.from_ros2_bag(self.path_hercules_bag, topic, output_folder)

        # Read the images directly from the rosbag for comparison
        typestore2 = Ros2BagWrapper._create_typestore_with_external_msgs(Stores.ROS2_HUMBLE, self.path_external_msgs_ros2)
        ros2_msgs = []
        with Reader2(self.path_hercules_bag) as reader2:
            connections2 = [x for x in reader2.connections if x.topic == topic]
            for conn2, timestamp2, rawdata2 in reader2.messages(connections=connections2):
                ros2_msgs.append(typestore2.deserialize_cdr(rawdata2, conn2.msgtype))

        # Make sure that there is at least one message to check
        np.testing.assert_raises(AssertionError, np.testing.assert_equal, len(ros2_msgs), 0)

        # Load the images and times from the .npy files
        images = np.load(file_path_imgs, mmap_mode='r')
        times = np.load(file_path_times)

        # Make sure the number of images between both sources match
        np.testing.assert_equal(len(images), len(ros2_msgs))

        # Assert that each image & time matches exactly
        for i in range(0, len(ros2_msgs)):
            msg2 = ros2_msgs[i]
            np.testing.assert_equal(times[i], msg2.header.stamp.sec + msg2.header.stamp.nanosec * 1e-9)
            np.testing.assert_array_equal(images[i], np.frombuffer(msg2.data, dtype=dtype).reshape(expected_shape))

    def test_from_ros2_bag_matches_raw_messages(self):
        """ Ensure both RGB and Depth imagery extracted via from_ros2_bag matches raw ROS2 messages. """
        self.helper_from_ros2_bag_matches_raw_messages("/hercules_node/Husky2/front_center_Scene/image", np.uint8, (720, 1280, 3))
        self.helper_from_ros2_bag_matches_raw_messages("/hercules_node/Husky2/front_center_DepthPlanar/image", np.float32, (720, 1280))

    def test_downscale_by_factor(self):
        """ Test image downscaling for Mono8 images. """
        # Create small Mono8 data (10x10)
        imgs = np.random.randint(0, 255, (2, 10, 10), dtype=np.uint8)
        data = ImageDataInMemory('cam', [0.0, 1.0], 10, 10, ImageDataInMemory.ImageEncoding.Mono8, imgs)

        data.downscale_by_factor(2)
        self.assertEqual(data.height, 5)
        self.assertEqual(data.width, 5)
        self.assertEqual(data.images.shape, (2, 5, 5))

    def test_downscale_by_factor_invalid_scale(self):
        """ Test that downscale_by_factor raises ValueError for invalid scale. """
        imgs = np.zeros((2, 10, 10), dtype=np.uint8)
        data = ImageDataInMemory('cam', [0.0, 1.0], 10, 10, ImageDataInMemory.ImageEncoding.Mono8, imgs)
        with self.assertRaises(ValueError):
            data.downscale_by_factor(3)  # 10 is not divisible by 3

    def test_downscale_by_factor_non_mono8_raises(self):
        """ Test that downscale_by_factor raises NotImplementedError for non-Mono8. """
        imgs = np.zeros((2, 10, 10, 3), dtype=np.uint8)
        data = ImageDataInMemory('cam', [0.0, 1.0], 10, 10, ImageDataInMemory.ImageEncoding.RGB8, imgs)
        with self.assertRaises(NotImplementedError):
            data.downscale_by_factor(2)




if __name__ == "__main__":
    unittest.main()
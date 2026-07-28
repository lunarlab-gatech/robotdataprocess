import decimal
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.CameraData import CameraData
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import BagLazyImageArray, ImageDataOnDisk, LazyImageArray
from rosbags.rosbag1 import Writer as Writer1
from rosbags.typesys import Stores, get_typestore
import tempfile
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
        
        # Test conversion from Mono8 to BGR8
        mono_folder = Path(Path('.'), 'tests', 'temporary_files', 'test_ImageDataOnDisk', 'mono_images').absolute()
        mono_folder.mkdir(parents=True, exist_ok=True)
        # Create a dummy mono image
        mono_image_path = mono_folder / "1.000000000.png"
        img = Image.new('L', (100, 100)) # 'L' mode for monochrome
        img.save(str(mono_image_path))

        mono_image_data = ImageDataOnDisk.from_image_files(mono_folder, 'optical')
        self.assertEqual(mono_image_data.encoding, ImageData.ImageEncoding.Mono8)
        original_mono_image = mono_image_data.images[0]

        mono_image_data.to_encoding(ImageData.ImageEncoding.BGR8)
        self.assertEqual(mono_image_data.encoding, ImageData.ImageEncoding.BGR8)
        self.assertEqual(len(mono_image_data.images.transformations), 1)

        bgr_from_mono = mono_image_data.images[0]
        self.assertEqual(bgr_from_mono.shape, (100, 100, 3))
        mono_converted_back = cv2.cvtColor(bgr_from_mono, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(original_mono_image, mono_converted_back)

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


    # =========================================================================
    # ====================== crop_images_to_LiDAR_FOV ========================
    # =========================================================================

    def _make_camera(self, width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0) -> CameraData:
        return CameraData.from_user_mono('cam', width, height, fx=fx, fy=fy, cx=cx, cy=cy)

    def test_crop_images_to_LiDAR_FOV_dimensions(self):
        """ Height and cy are updated correctly after cropping. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        img_data = ImageDataOnDisk.from_image_files(folder_path, 'cam')
        cam = self._make_camera(width=img_data.width, height=img_data.height)

        fy, cy = float(cam.K[1, 1]), float(cam.K[1, 2])
        lidar_v_fov = (-10.0, 10.0)
        img_data.crop_images_to_LiDAR_FOV(lidar_v_fov, cam)

        expected_row_top    = max(0, int(np.floor(cy - fy * np.tan(np.radians(lidar_v_fov[1])))))
        expected_row_bottom = min(img_data.height + expected_row_top,
                                  int(np.ceil(cy - fy * np.tan(np.radians(lidar_v_fov[0])))))
        expected_height = expected_row_bottom - expected_row_top
        expected_cy = cy - expected_row_top

        self.assertEqual(img_data.height, expected_height)
        self.assertEqual(cam.height, expected_height)
        self.assertAlmostEqual(float(cam.K[1, 2]), expected_cy)
        self.assertAlmostEqual(float(cam.P[1, 2]), expected_cy)

    def test_crop_images_to_LiDAR_FOV_image_shape(self):
        """ Loaded images have the cropped row count after crop_images_to_LiDAR_FOV. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        img_data = ImageDataOnDisk.from_image_files(folder_path, 'cam')
        cam = self._make_camera(width=img_data.width, height=img_data.height)

        img_data.crop_images_to_LiDAR_FOV((-10.0, 10.0), cam)

        for i in range(img_data.len()):
            self.assertEqual(img_data.images[i].shape[0], img_data.height)

    def test_crop_images_to_LiDAR_FOV_pixel_content(self):
        """ Cropped images contain exactly the rows [row_top:row_bottom] of the originals. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        orig = ImageDataOnDisk.from_image_files(folder_path, 'cam')
        cropped = ImageDataOnDisk.from_image_files(folder_path, 'cam')
        cam = self._make_camera(width=orig.width, height=orig.height)

        fy, cy = float(cam.K[1, 1]), float(cam.K[1, 2])
        lidar_v_fov = (-10.0, 10.0)
        row_top    = max(0, int(np.floor(cy - fy * np.tan(np.radians(lidar_v_fov[1])))))
        row_bottom = min(orig.height, int(np.ceil(cy - fy * np.tan(np.radians(lidar_v_fov[0])))))

        cropped.crop_images_to_LiDAR_FOV(lidar_v_fov, cam)

        for i in range(orig.len()):
            np.testing.assert_array_equal(cropped.images[i], orig.images[i][row_top:row_bottom, :])

    def test_crop_images_to_LiDAR_FOV_invalid_fov_raises(self):
        """ min >= max raises ValueError. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        img_data = ImageDataOnDisk.from_image_files(folder_path, 'cam')
        cam = self._make_camera(width=img_data.width, height=img_data.height)

        with self.assertRaises(ValueError):
            img_data.crop_images_to_LiDAR_FOV((10.0, 10.0), cam)
        with self.assertRaises(ValueError):
            img_data.crop_images_to_LiDAR_FOV((15.0, 10.0), cam)

    def test_crop_images_to_LiDAR_FOV_no_camera_raises(self):
        """ Omitting camera_data raises ValueError. """
        folder_path = Path(Path('.'), 'tests', 'files', 'test_ImageDataOnDisk', 'test_from_image_files').absolute()
        img_data = ImageDataOnDisk.from_image_files(folder_path, 'cam')

        with self.assertRaises((ValueError, TypeError)):
            img_data.crop_images_to_LiDAR_FOV((-10.0, 10.0))


    def _write_ros1_image_bag(self, bag_path: Path, topic: str, frame_id: str,
                               images: np.ndarray, timestamps_sec: list) -> None:
        """Write a ROS1 bag containing sensor_msgs/msg/Image messages."""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        ImageMsg = typestore.types['sensor_msgs/msg/Image']
        Header    = typestore.types['std_msgs/msg/Header']
        Time      = typestore.types['builtin_interfaces/msg/Time']

        n, H, W = images.shape[:3]
        channels = 1 if images.ndim == 3 else images.shape[3]
        encoding = 'rgb8' if channels == 3 else 'mono8'
        step = W * channels

        with Writer1(bag_path) as writer:
            conn = writer.add_connection(topic, ImageMsg.__msgtype__, typestore=typestore)
            for i, ts in enumerate(timestamps_sec):
                ts_dec = Decimal(str(ts))
                sec  = int(ts_dec)
                nsec = int((ts_dec - Decimal(sec)) * Decimal('1e9'))
                ts_ns = sec * 10**9 + nsec
                msg = ImageMsg(
                    Header(seq=i, stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id),
                    height=H, width=W, encoding=encoding,
                    is_bigendian=0, step=step,
                    data=images[i].flatten(),
                )
                writer.write(conn, ts_ns, typestore.serialize_ros1(msg, ImageMsg.__msgtype__))

    def test_from_ros1_bag(self):
        """Write a ROS1 bag with known images and verify from_ros1_bag round-trips correctly."""
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0'
        timestamps_sec = [1.0, 2.0, 3.0]

        # Three small RGB8 images with distinct solid colours
        images = np.zeros((3, H, W, 3), dtype=np.uint8)
        images[0, :, :] = [255,   0,   0]  # red
        images[1, :, :] = [  0, 255,   0]  # green
        images[2, :, :] = [  0,   0, 255]  # blue

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test.bag'
            self._write_ros1_image_bag(bag_path, topic, frame_id, images, timestamps_sec)

            data = ImageDataOnDisk.from_ros1_bag(bag_path, topic)

            # --- metadata ---
            self.assertEqual(data.frame_id, frame_id)
            self.assertEqual(data.height, H)
            self.assertEqual(data.width, W)
            self.assertEqual(data.encoding, ImageData.ImageEncoding.RGB8)
            self.assertEqual(data.len(), 3)
            self.assertIsInstance(data.images, BagLazyImageArray)

            # --- timestamps ---
            np.testing.assert_array_almost_equal(
                data.timestamps.astype(np.float64), timestamps_sec, decimal=6)

            # --- pixel data loaded on demand ---
            for i in range(3):
                np.testing.assert_array_equal(data.images[i], images[i])

            # --- crop_data keeps the right subset ---
            data2 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            data2.crop_data(Decimal('1.5'), Decimal('3.0'))
            self.assertEqual(data2.len(), 2)
            np.testing.assert_array_equal(data2.images[0], images[1])  # green
            np.testing.assert_array_equal(data2.images[1], images[2])  # blue

            # --- to_encoding RGB -> BGR ---
            data3 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            data3.to_encoding(ImageData.ImageEncoding.BGR8)
            self.assertEqual(data3.encoding, ImageData.ImageEncoding.BGR8)
            expected_bgr_red = np.zeros((H, W, 3), dtype=np.uint8)
            expected_bgr_red[:, :] = [0, 0, 255]  # red in BGR
            np.testing.assert_array_equal(data3.images[0], expected_bgr_red)

            # --- crop_images_to_LiDAR_FOV ---
            # With H=4, cy=2.0, fy=10.0, FOV=(-5°,5°):
            #   row_top    = floor(2.0 - 10.0*tan(5°))  = 1
            #   row_bottom = ceil (2.0 - 10.0*tan(-5°)) = 3  →  new_height = 2
            data4 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            cam = CameraData.from_user_mono('test_cam', W, H, fx=10.0, fy=10.0, cx=3.0, cy=2.0)
            fy_val, cy_val = float(cam.K[1, 1]), float(cam.K[1, 2])
            lidar_v_fov = (-5.0, 5.0)
            row_top    = max(0, int(np.floor(cy_val - fy_val * np.tan(np.radians(lidar_v_fov[1])))))
            row_bottom = min(H,  int(np.ceil( cy_val - fy_val * np.tan(np.radians(lidar_v_fov[0])))))
            data4.crop_images_to_LiDAR_FOV(lidar_v_fov, cam)
            self.assertEqual(data4.height, row_bottom - row_top)
            self.assertEqual(cam.height, row_bottom - row_top)
            for i in range(3):
                np.testing.assert_array_equal(data4.images[i], images[i][row_top:row_bottom, :])

    def test_from_ros1_bag_mono8_to_encoding_bgr8(self):
        """Write a ROS1 bag with mono8 images and verify from_ros1_bag + to_encoding(BGR8) round-trips."""
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0'
        timestamps_sec = [1.0, 2.0, 3.0]

        # Three small Mono8 images with distinct solid grey levels
        images = np.zeros((3, H, W), dtype=np.uint8)
        images[0, :, :] = 10
        images[1, :, :] = 128
        images[2, :, :] = 250

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test.bag'
            self._write_ros1_image_bag(bag_path, topic, frame_id, images, timestamps_sec)

            data = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            self.assertEqual(data.encoding, ImageData.ImageEncoding.Mono8)
            for i in range(3):
                np.testing.assert_array_equal(data.images[i], images[i])

            data.to_encoding(ImageData.ImageEncoding.BGR8)
            self.assertEqual(data.encoding, ImageData.ImageEncoding.BGR8)
            for i in range(3):
                bgr_image = data.images[i]
                self.assertEqual(bgr_image.shape, (H, W, 3))
                mono_converted_back = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                np.testing.assert_array_equal(images[i], mono_converted_back)

    def test_crop_to_matched(self):
        """ crop_to_matched keeps .images in sync with .timestamps for both BagLazyImageArray-backed objects. """
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0'
        timestamps_sec = [1.0, 2.0, 3.0]

        images = np.zeros((3, H, W, 3), dtype=np.uint8)
        images[0, :, :] = [255,   0,   0]  # red
        images[1, :, :] = [  0, 255,   0]  # green
        images[2, :, :] = [  0,   0, 255]  # blue

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test.bag'
            self._write_ros1_image_bag(bag_path, topic, frame_id, images, timestamps_sec)

            data1 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            data2 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)

            # data2 is missing the middle (green) entry, and its last timestamp is
            # slightly offset from data1's, but still within tolerance.
            data2.timestamps = np.array([data2.timestamps[0], data2.timestamps[2] + Decimal('0.02')])
            data2.images = data2.images[np.array([True, False, True])]

            ImageDataOnDisk.crop_to_matched(data1, data2, Decimal('0.05'))

            self.assertEqual(data1.len(), 2)
            self.assertEqual(data2.len(), 2)
            np.testing.assert_array_equal(data1.images[0], images[0])  # red
            np.testing.assert_array_equal(data1.images[1], images[2])  # blue
            np.testing.assert_array_equal(data2.images[0], images[0])  # red
            np.testing.assert_array_equal(data2.images[1], images[2])  # blue

    def _write_ros1_compressed_image_bag(self, bag_path: Path, topic: str, frame_id: str,
                                          images: np.ndarray, timestamps_sec: list,
                                          format_str: str = 'bgr8; png compressed bgr8') -> None:
        """Write a ROS1 bag containing sensor_msgs/msg/CompressedImage messages (PNG)."""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        CompressedImageMsg = typestore.types['sensor_msgs/msg/CompressedImage']
        Header = typestore.types['std_msgs/msg/Header']
        Time   = typestore.types['builtin_interfaces/msg/Time']

        with Writer1(bag_path) as writer:
            conn = writer.add_connection(topic, CompressedImageMsg.__msgtype__, typestore=typestore)
            for i, ts in enumerate(timestamps_sec):
                ts_dec = Decimal(str(ts))
                sec  = int(ts_dec)
                nsec = int((ts_dec - Decimal(sec)) * Decimal('1e9'))
                ts_ns = sec * 10**9 + nsec
                ok, buf = cv2.imencode('.png', images[i])
                assert ok, "cv2.imencode failed"
                msg = CompressedImageMsg(
                    Header(seq=i, stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id),
                    format=format_str,
                    data=buf.flatten(),
                )
                writer.write(conn, ts_ns, typestore.serialize_ros1(msg, CompressedImageMsg.__msgtype__))

    def test_from_ros1_bag_compressed(self):
        """Write a ROS1 bag with compressed images and verify from_ros1_bag round-trips correctly."""
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0/compressed'
        timestamps_sec = [1.0, 2.0, 3.0]

        # Three small BGR8 images with distinct solid colours (cv2 convention)
        images = np.zeros((3, H, W, 3), dtype=np.uint8)
        images[0, :, :] = [255,   0,   0]  # blue channel
        images[1, :, :] = [  0, 255,   0]  # green channel
        images[2, :, :] = [  0,   0, 255]  # red channel

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test_compressed.bag'
            self._write_ros1_compressed_image_bag(bag_path, topic, frame_id, images, timestamps_sec)

            data = ImageDataOnDisk.from_ros1_bag(bag_path, topic)

            # --- metadata ---
            self.assertEqual(data.frame_id, frame_id)
            self.assertEqual(data.height, H)
            self.assertEqual(data.width, W)
            self.assertEqual(data.encoding, ImageData.ImageEncoding.BGR8)
            self.assertEqual(data.len(), 3)
            self.assertIsInstance(data.images, BagLazyImageArray)

            # --- timestamps ---
            np.testing.assert_array_almost_equal(
                data.timestamps.astype(np.float64), timestamps_sec, decimal=6)

            # --- pixel data round-trips losslessly through PNG ---
            for i in range(3):
                np.testing.assert_array_equal(data.images[i], images[i])

            # --- crop_data keeps the right subset ---
            data2 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            data2.crop_data(Decimal('1.5'), Decimal('3.0'))
            self.assertEqual(data2.len(), 2)
            np.testing.assert_array_equal(data2.images[0], images[1])
            np.testing.assert_array_equal(data2.images[1], images[2])

            # --- crop_images_to_LiDAR_FOV ---
            data3 = ImageDataOnDisk.from_ros1_bag(bag_path, topic)
            cam = CameraData.from_user_mono('test_cam', W, H, fx=10.0, fy=10.0, cx=3.0, cy=2.0)
            fy_val, cy_val = float(cam.K[1, 1]), float(cam.K[1, 2])
            lidar_v_fov = (-5.0, 5.0)
            row_top    = max(0, int(np.floor(cy_val - fy_val * np.tan(np.radians(lidar_v_fov[1])))))
            row_bottom = min(H,  int(np.ceil( cy_val - fy_val * np.tan(np.radians(lidar_v_fov[0])))))
            data3.crop_images_to_LiDAR_FOV(lidar_v_fov, cam)
            self.assertEqual(data3.height, row_bottom - row_top)
            self.assertEqual(cam.height, row_bottom - row_top)
            for i in range(3):
                np.testing.assert_array_equal(data3.images[i], images[i][row_top:row_bottom, :])


    def _write_ros1_image_bag_skewed_timestamps(self, bag_path: Path, topic: str, frame_id: str,
                                                  images: np.ndarray, header_timestamps_sec: list,
                                                  recording_timestamps_sec: list) -> None:
        """Write a ROS1 bag where bag recording times differ from msg.header.stamp values."""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        ImageMsg = typestore.types['sensor_msgs/msg/Image']
        Header    = typestore.types['std_msgs/msg/Header']
        Time      = typestore.types['builtin_interfaces/msg/Time']

        n, H, W = images.shape[:3]
        channels = 1 if images.ndim == 3 else images.shape[3]
        encoding = 'rgb8' if channels == 3 else 'mono8'
        step = W * channels

        with Writer1(bag_path) as writer:
            conn = writer.add_connection(topic, ImageMsg.__msgtype__, typestore=typestore)
            for i, (h_ts, r_ts) in enumerate(zip(header_timestamps_sec, recording_timestamps_sec)):
                h_dec = Decimal(str(h_ts))
                h_sec  = int(h_dec)
                h_nsec = int((h_dec - Decimal(h_sec)) * Decimal('1e9'))

                r_dec = Decimal(str(r_ts))
                r_sec  = int(r_dec)
                r_nsec = int((r_dec - Decimal(r_sec)) * Decimal('1e9'))
                rec_ns = r_sec * 10**9 + r_nsec

                msg = ImageMsg(
                    Header(seq=i, stamp=Time(sec=h_sec, nanosec=h_nsec), frame_id=frame_id),
                    height=H, width=W, encoding=encoding,
                    is_bigendian=0, step=step,
                    data=images[i].flatten(),
                )
                writer.write(conn, rec_ns, typestore.serialize_ros1(msg, ImageMsg.__msgtype__))

    def test_from_ros1_bag_uses_header_stamp(self):
        """from_ros1_bag must load timestamps from msg.header.stamp, not bag recording time.

        Writes a bag where recording times are ~1000 s ahead of the header stamps.
        The bug (entry.time used instead of msg.header.stamp) would cause timestamps
        to be [1001, 1002, 1003] instead of the correct [1, 2, 3].  The fix must also
        ensure images are still retrievable after the seek logic is corrected.
        """
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0'
        header_timestamps_sec   = [1.0, 2.0, 3.0]
        recording_timestamps_sec = [1001.0, 1002.0, 1003.0]

        images = np.zeros((3, H, W, 3), dtype=np.uint8)
        images[0, :, :] = [255,   0,   0]
        images[1, :, :] = [  0, 255,   0]
        images[2, :, :] = [  0,   0, 255]

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test_skewed.bag'
            self._write_ros1_image_bag_skewed_timestamps(
                bag_path, topic, frame_id, images, header_timestamps_sec, recording_timestamps_sec)

            data = ImageDataOnDisk.from_ros1_bag(bag_path, topic)

            # Timestamps must come from msg.header.stamp, not the bag recording time
            np.testing.assert_array_almost_equal(
                data.timestamps.astype(np.float64), header_timestamps_sec, decimal=6)

            # Images must still load correctly after the seek logic is corrected
            for i in range(3):
                np.testing.assert_array_equal(data.images[i], images[i])

    def test_from_ros1_bag_header_recording_order_mismatch(self):
        """from_ros1_bag must return images ordered by header stamp even when recording
        time order differs from header stamp order.

        Bag layout (recording-time order):
          rec=1001s  header=3.0s  red image
          rec=1002s  header=1.0s  green image
          rec=1003s  header=2.0s  blue image

        Expected after from_ros1_bag (sorted by header stamp):
          timestamps = [1.0, 2.0, 3.0]
          images[0]  = green  (header 1.0s, rec 1002s)
          images[1]  = blue   (header 2.0s, rec 1003s)
          images[2]  = red    (header 3.0s, rec 1001s)

        This catches a bug where seek uses header-stamp values as recording-time
        indices, which would retrieve the wrong message when the two orderings differ.
        """
        H, W = 4, 6
        frame_id = 'test_cam'
        topic = '/cam0'

        # Images written in recording-time order
        images_written = np.zeros((3, H, W, 3), dtype=np.uint8)
        images_written[0, :, :] = [255,   0,   0]  # red   → header 3.0s
        images_written[1, :, :] = [  0, 255,   0]  # green → header 1.0s
        images_written[2, :, :] = [  0,   0, 255]  # blue  → header 2.0s

        # Recording times ascending (required by bag format); header stamps out of order
        recording_timestamps_sec = [1001.0, 1002.0, 1003.0]
        header_timestamps_sec    = [   3.0,    1.0,    2.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'test_order_mismatch.bag'
            self._write_ros1_image_bag_skewed_timestamps(
                bag_path, topic, frame_id, images_written,
                header_timestamps_sec, recording_timestamps_sec)

            data = ImageDataOnDisk.from_ros1_bag(bag_path, topic)

            # Timestamps must be sorted by header stamp
            np.testing.assert_array_almost_equal(
                data.timestamps.astype(np.float64), [1.0, 2.0, 3.0], decimal=6)

            # images[i] must match the image whose header stamp is timestamps[i]
            np.testing.assert_array_equal(data.images[0], images_written[1])  # green (header 1.0s)
            np.testing.assert_array_equal(data.images[1], images_written[2])  # blue  (header 2.0s)
            np.testing.assert_array_equal(data.images[2], images_written[0])  # red   (header 3.0s)


if __name__ == "__main__":
    unittest.main()
import matplotlib
matplotlib.use('Agg')
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
import tempfile
import unittest

from robotdataprocess.data_types.CameraData import CameraData
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from rosbags.rosbag1 import Writer as Writer1
from rosbags.typesys import Stores, get_typestore


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestCameraData(unittest.TestCase):

    # Shared calibration values used across tests
    FX, FY = 500.0, 500.0
    CX, CY = 320.0, 240.0
    WIDTH, HEIGHT = 640, 480
    FRAME_ID = "camera_optical_frame"
    D = [0.1, -0.2, 0.0, 0.0, 0.05]

    def _make_camera(self, D=None, timeshift_cam_imu=0.0) -> CameraData:
        return CameraData.from_user_mono(
            frame_id=self.FRAME_ID,
            width=self.WIDTH,
            height=self.HEIGHT,
            fx=self.FX, fy=self.FY,
            cx=self.CX, cy=self.CY,
            timeshift_cam_imu=timeshift_cam_imu,
            D=D)

    # =========================================================================
    # ========================= DistortionModel enum ==========================
    # =========================================================================

    def test_distortion_model_to_ros_str(self):
        """ RADIAL_TANGENTIAL maps to 'plumb_bob'. """
        self.assertEqual(
            CameraData.DistortionModel.to_ros_str(CameraData.DistortionModel.RADIAL_TANGENTIAL),
            "plumb_bob")

    def test_distortion_model_from_ros_str(self):
        """ 'plumb_bob' maps back to RADIAL_TANGENTIAL. """
        self.assertEqual(
            CameraData.DistortionModel.from_ros_str("plumb_bob"),
            CameraData.DistortionModel.RADIAL_TANGENTIAL)

    def test_distortion_model_from_ros_str_unknown_raises(self):
        """ Unknown ROS distortion string raises NotImplementedError. """
        with self.assertRaises(NotImplementedError):
            CameraData.DistortionModel.from_ros_str("kannala_brandt")

    # =========================================================================
    # ============================ CameraModel enum ============================
    # =========================================================================

    def test_camera_model_from_kalibr_str(self):
        """ 'pinhole' maps to CameraModel.PINHOLE. """
        self.assertEqual(
            CameraData.CameraModel.from_kalibr_str("pinhole"),
            CameraData.CameraModel.PINHOLE)

    def test_camera_model_from_kalibr_str_unsupported_raises(self):
        """ Unknown kalibr camera model string raises NotImplementedError. """
        with self.assertRaises(NotImplementedError):
            CameraData.CameraModel.from_kalibr_str("omnidirectional")

    # =========================================================================
    # =========================== from_user_mono ==============================
    # =========================================================================

    def test_from_user_mono_K(self):
        """ K is assembled correctly from fx, fy, cx, cy. """
        cam = self._make_camera()
        expected_K = np.array([[self.FX, 0.0, self.CX],
                                [0.0, self.FY, self.CY],
                                [0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(cam.K, expected_K)

    def test_from_user_mono_R_is_identity(self):
        """ R is always the identity matrix for monocular cameras. """
        cam = self._make_camera()
        np.testing.assert_array_equal(cam.R, np.eye(3))

    def test_from_user_mono_P_derived_from_K(self):
        """ P is [K | 0], i.e. K with a zero fourth column appended. """
        cam = self._make_camera()
        expected_P = np.zeros((3, 4))
        expected_P[:3, :3] = cam.K
        np.testing.assert_array_equal(cam.P, expected_P)

    def test_from_user_mono_D_default_zeros(self):
        """ Default D is five zeros when not provided. """
        cam = self._make_camera()
        np.testing.assert_array_equal(cam.D, np.zeros(5))

    def test_from_user_mono_D_custom(self):
        """ Custom D values are stored correctly. """
        cam = self._make_camera(D=self.D)
        np.testing.assert_array_equal(cam.D, np.array(self.D))

    def test_from_user_mono_metadata(self):
        """ frame_id, width, height, distortion_model, and camera_model are stored correctly. """
        cam = self._make_camera()
        self.assertEqual(cam.frame_id, self.FRAME_ID)
        self.assertEqual(cam.width, self.WIDTH)
        self.assertEqual(cam.height, self.HEIGHT)
        self.assertEqual(cam.distortion_model, CameraData.DistortionModel.RADIAL_TANGENTIAL)
        self.assertEqual(cam.camera_model, CameraData.CameraModel.PINHOLE)

    def test_from_user_mono_timeshift_cam_imu_default(self):
        """ timeshift_cam_imu defaults to 0. """
        cam = self._make_camera()
        self.assertEqual(cam.timeshift_cam_imu, 0.0)

    def test_from_user_mono_timeshift_cam_imu_custom(self):
        """ Custom timeshift_cam_imu is stored correctly. """
        cam = self._make_camera(timeshift_cam_imu=0.0456)
        self.assertEqual(cam.timeshift_cam_imu, 0.0456)

    # =========================================================================
    # ========================= __init__ validation ===========================
    # =========================================================================

    def test_invalid_width_raises(self):
        """ Non-positive width raises ValueError. """
        with self.assertRaises(ValueError):
            CameraData(frame_id=self.FRAME_ID, width=0, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       camera_model=CameraData.CameraModel.PINHOLE,
                       timeshift_cam_imu=0.0,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_invalid_height_raises(self):
        """ Non-positive height raises ValueError. """
        with self.assertRaises(ValueError):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=-1,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       camera_model=CameraData.CameraModel.PINHOLE,
                       timeshift_cam_imu=0.0,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_K_wrong_shape_raises(self):
        """ K with wrong shape raises an error. """
        with self.assertRaises(Exception):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       camera_model=CameraData.CameraModel.PINHOLE,
                       timeshift_cam_imu=0.0,
                       K=np.eye(4), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_P_wrong_shape_raises(self):
        """ P with wrong shape raises an error. """
        with self.assertRaises(Exception):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       camera_model=CameraData.CameraModel.PINHOLE,
                       timeshift_cam_imu=0.0,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 3)))

    # =========================================================================
    # ============================ get_ros_msg ================================
    # =========================================================================

    def test_get_ros_msg_rosbags_fields(self):
        """ ROSBAGS message has correct header, dims, distortion model, and matrices. """
        cam = self._make_camera(D=self.D)
        msg = cam.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)

        self.assertEqual(msg.header.frame_id, self.FRAME_ID)
        self.assertEqual(msg.header.stamp.sec, 0)
        self.assertEqual(msg.header.stamp.nanosec, 0)
        self.assertEqual(msg.width, self.WIDTH)
        self.assertEqual(msg.height, self.HEIGHT)
        self.assertEqual(msg.distortion_model, "plumb_bob")
        np.testing.assert_array_almost_equal(msg.d, np.array(self.D))
        np.testing.assert_array_almost_equal(msg.k, cam.K.flatten())
        np.testing.assert_array_almost_equal(msg.r, cam.R.flatten())
        np.testing.assert_array_almost_equal(msg.p, cam.P.flatten())

    def test_get_ros_msg_rosbags_binning_and_roi(self):
        """ ROSBAGS message has binning and ROI zeroed out. """
        cam = self._make_camera()
        msg = cam.get_ros_msg(ROSMsgLibType.ROSBAGS, 0)

        self.assertEqual(msg.binning_x, 0)
        self.assertEqual(msg.binning_y, 0)
        self.assertEqual(msg.roi.x_offset, 0)
        self.assertEqual(msg.roi.y_offset, 0)
        self.assertEqual(msg.roi.width, 0)
        self.assertEqual(msg.roi.height, 0)
        self.assertFalse(msg.roi.do_rectify)

    def test_get_ros_msg_none_raises(self):
        """ NONE lib type raises NotImplementedError. """
        cam = self._make_camera()
        with self.assertRaises(NotImplementedError):
            cam.get_ros_msg(ROSMsgLibType.NONE, 0)

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    # =========================================================================
    # =========================== from_kalibr_mono ============================
    # =========================================================================

    KALIBR_YAML = Path(Path('.'), 'tests', 'files', 'test_CameraData', 'test_from_kalibr_mon', 'stereo.yaml').absolute()

    def test_from_kalibr_mono_cam0_intrinsics(self):
        """ cam0 K matrix is assembled correctly from kalibr intrinsics. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        expected_K = np.array([[940.862825677534,  0.0,               799.1626975233576],
                                [0.0,               938.554923506332,  559.295406893583 ],
                                [0.0,               0.0,               1.0              ]])
        np.testing.assert_array_almost_equal(cam.K, expected_K)

    def test_from_kalibr_mono_cam0_distortion(self):
        """ cam0 D vector matches kalibr distortion_coeffs. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        expected_D = np.array([-0.1008504099655989, 0.08905706623788286,
                                0.0007516966627205781, -0.0011958374307601393])
        np.testing.assert_array_almost_equal(cam.D, expected_D)

    def test_from_kalibr_mono_cam0_resolution(self):
        """ cam0 width and height match kalibr resolution. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        self.assertEqual(cam.width, 1600)
        self.assertEqual(cam.height, 1100)

    def test_from_kalibr_mono_cam0_R_and_P(self):
        """ R is identity and P is [K|0] for a monocular load. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        np.testing.assert_array_equal(cam.R, np.eye(3))
        expected_P = np.zeros((3, 4))
        expected_P[:3, :3] = cam.K
        np.testing.assert_array_almost_equal(cam.P, expected_P)

    def test_from_kalibr_mono_cam0_frame_id_and_model(self):
        """ frame_id is the camera name and distortion/camera models are correct. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        self.assertEqual(cam.frame_id, 'cam0')
        self.assertEqual(cam.distortion_model, CameraData.DistortionModel.RADIAL_TANGENTIAL)
        self.assertEqual(cam.camera_model, CameraData.CameraModel.PINHOLE)

    def test_from_kalibr_mono_cam1_intrinsics(self):
        """ cam1 K matrix is assembled correctly from kalibr intrinsics. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam1')
        expected_K = np.array([[934.5190744321391,  0.0,               792.8073165035943],
                                [0.0,               932.525429508503,  562.7061769000949],
                                [0.0,               0.0,               1.0             ]])
        np.testing.assert_array_almost_equal(cam.K, expected_K)

    def test_from_kalibr_mono_cam0_timeshift_cam_imu_defaults_to_zero(self):
        """ cam0 has no timeshift_cam_imu key, so it defaults to 0. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam0')
        self.assertEqual(cam.timeshift_cam_imu, 0.0)

    def test_from_kalibr_mono_cam1_timeshift_cam_imu_loaded(self):
        """ cam1 timeshift_cam_imu is loaded from the YAML key. """
        cam = CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam1')
        self.assertEqual(cam.timeshift_cam_imu, 0.0123)

    def test_from_kalibr_mono_missing_camera_raises(self):
        """ Requesting a camera not in the YAML raises KeyError. """
        with self.assertRaises(KeyError):
            CameraData.from_kalibr_mono(self.KALIBR_YAML, 'cam99')

    def test_from_kalibr_str_unsupported_raises(self):
        """ from_kalibr_str raises NotImplementedError for unknown models. """
        with self.assertRaises(NotImplementedError):
            CameraData.DistortionModel.from_kalibr_str('equidistant')

    # =========================================================================
    # ============================ visualize_FOV ==============================
    # =========================================================================

    def test_visualize_FOV_camera_only(self):
        """ visualize_FOV with no LiDAR overlay does not crash. """
        cam = self._make_camera()
        cam.visualize_FOV(depth=5.0, testing=True)

    def test_visualize_FOV_with_lidar(self):
        """ visualize_FOV with a LiDAR vertical FOV overlay does not crash. """
        cam = self._make_camera()
        cam.visualize_FOV(depth=5.0, lidar_v_fov=(-15.0, 15.0), testing=True)

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def test_sync_to_ImageData(self):
        """ sync_to_ImageData copies timestamps from ImageDataInMemory exactly. """
        timestamps = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]
        images = np.zeros((3, self.HEIGHT, self.WIDTH), dtype=np.uint8)
        image_data = ImageDataInMemory(
            frame_id=self.FRAME_ID,
            timestamps=timestamps,
            height=self.HEIGHT,
            width=self.WIDTH,
            encoding=ImageData.ImageEncoding.Mono8,
            images=images)

        cam = self._make_camera()
        cam.sync_to_ImageData(image_data)

        np.testing.assert_array_equal(cam.timestamps, np.array(timestamps))


    # =========================================================================
    # ============================ from_ros1_bag ==============================
    # =========================================================================

    def _write_ros1_camera_info_bag(self, bag_path: Path, topic: str, frame_id: str,
                                    width: int, height: int, distortion_model: str,
                                    K: np.ndarray, D: np.ndarray,
                                    R: np.ndarray, P: np.ndarray) -> None:
        """Write a ROS1 bag containing a single sensor_msgs/CameraInfo message."""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        CameraInfoMsg = typestore.types['sensor_msgs/msg/CameraInfo']
        Header = typestore.types['std_msgs/msg/Header']
        Time = typestore.types['builtin_interfaces/msg/Time']
        RegionOfInterest = typestore.types['sensor_msgs/msg/RegionOfInterest']

        msg = CameraInfoMsg(
            header=Header(seq=0, stamp=Time(sec=1, nanosec=0), frame_id=frame_id),
            height=height,
            width=width,
            distortion_model=distortion_model,
            D=D,
            K=K.flatten(),
            R=R.flatten(),
            P=P.flatten(),
            binning_x=0,
            binning_y=0,
            roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
        )

        with Writer1(bag_path) as writer:
            conn = writer.add_connection(topic, CameraInfoMsg.__msgtype__, typestore=typestore)
            writer.write(conn, 1_000_000_000, typestore.serialize_ros1(msg, CameraInfoMsg.__msgtype__))

    def test_from_ros1_bag(self):
        """Write a ROS1 bag with a CameraInfo message and verify from_ros1_bag round-trips."""
        topic = '/camera/camera_info'
        frame_id = 'camera_optical_frame'
        width, height = 640, 480
        distortion_model_str = 'plumb_bob'
        D = np.array([0.1, -0.2, 0.003, -0.0015, 0.05])
        K = np.array([[940.86, 0.0,    799.16],
                      [0.0,    938.55, 559.29],
                      [0.0,    0.0,    1.0   ]])
        R = np.array([[ 0.99998,  0.00412, -0.00431],
                      [-0.00413,  0.99999, -0.00180],
                      [ 0.00430,  0.00182,  0.99999]])
        P = np.array([[920.14,   0.0,    801.33,  -55.208],
                      [  0.0,   935.71,  561.05,    0.0  ],
                      [  0.0,     0.0,     1.0,     0.0  ]])

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'camera_info.bag'
            self._write_ros1_camera_info_bag(bag_path, topic, frame_id, width, height,
                                             distortion_model_str, K, D, R, P)

            cam = CameraData.from_ros1_bag(bag_path, topic)

            self.assertEqual(cam.frame_id, frame_id)
            self.assertEqual(cam.width, width)
            self.assertEqual(cam.height, height)
            self.assertEqual(cam.distortion_model, CameraData.DistortionModel.RADIAL_TANGENTIAL)
            self.assertEqual(cam.camera_model, CameraData.CameraModel.PINHOLE)
            self.assertEqual(cam.timeshift_cam_imu, 0.0)
            np.testing.assert_array_almost_equal(cam.K, K)
            np.testing.assert_array_almost_equal(cam.D, D)
            np.testing.assert_array_almost_equal(cam.R, R)
            np.testing.assert_array_almost_equal(cam.P, P)

            cam_shifted = CameraData.from_ros1_bag(bag_path, topic, timeshift_cam_imu=0.0789)
            self.assertEqual(cam_shifted.timeshift_cam_imu, 0.0789)

            with self.assertRaises(ValueError):
                CameraData.from_ros1_bag(bag_path, '/nonexistent_topic')

    # =========================================================================
    # ========================= Multi Data Methods =============================
    # =========================================================================

    def test_align_ImageData_and_CameraData_to_imu_ts(self):
        """ ImageData timestamps are shifted by timeshift_cam_imu, and it is then reset to 0. """
        timestamps = [Decimal('1.0'), Decimal('2.0'), Decimal('3.0')]
        images = np.zeros((3, self.HEIGHT, self.WIDTH), dtype=np.uint8)
        image_data = ImageDataInMemory(
            frame_id=self.FRAME_ID,
            timestamps=timestamps,
            height=self.HEIGHT,
            width=self.WIDTH,
            encoding=ImageData.ImageEncoding.Mono8,
            images=images)

        cam = self._make_camera(timeshift_cam_imu=0.5)

        CameraData.align_ImageData_and_CameraData_to_imu_ts(image_data, cam)

        expected_timestamps = np.array([Decimal('1.5'), Decimal('2.5'), Decimal('3.5')])
        np.testing.assert_array_equal(image_data.timestamps, expected_timestamps)
        self.assertEqual(cam.timeshift_cam_imu, 0.0)


if __name__ == "__main__":
    unittest.main()

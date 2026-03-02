import numpy as np
import os
import unittest

from robotdataprocess.data_types.CameraData import CameraData
from robotdataprocess.data_types.Data import ROSMsgLibType


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestCameraData(unittest.TestCase):

    # Shared calibration values used across tests
    FX, FY = 500.0, 500.0
    CX, CY = 320.0, 240.0
    WIDTH, HEIGHT = 640, 480
    FRAME_ID = "camera_optical_frame"
    D = [0.1, -0.2, 0.0, 0.0, 0.05]

    def _make_camera(self, D=None) -> CameraData:
        return CameraData.from_user_mono(
            frame_id=self.FRAME_ID,
            width=self.WIDTH,
            height=self.HEIGHT,
            fx=self.FX, fy=self.FY,
            cx=self.CX, cy=self.CY,
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
        """ frame_id, width, height, and distortion_model are stored correctly. """
        cam = self._make_camera()
        self.assertEqual(cam.frame_id, self.FRAME_ID)
        self.assertEqual(cam.width, self.WIDTH)
        self.assertEqual(cam.height, self.HEIGHT)
        self.assertEqual(cam.distortion_model, CameraData.DistortionModel.RADIAL_TANGENTIAL)

    # =========================================================================
    # ========================= __init__ validation ===========================
    # =========================================================================

    def test_invalid_width_raises(self):
        """ Non-positive width raises ValueError. """
        with self.assertRaises(ValueError):
            CameraData(frame_id=self.FRAME_ID, width=0, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_invalid_height_raises(self):
        """ Non-positive height raises ValueError. """
        with self.assertRaises(ValueError):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=-1,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_K_wrong_shape_raises(self):
        """ K with wrong shape raises an error. """
        with self.assertRaises(Exception):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       K=np.eye(4), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 4)))

    def test_P_wrong_shape_raises(self):
        """ P with wrong shape raises an error. """
        with self.assertRaises(Exception):
            CameraData(frame_id=self.FRAME_ID, width=self.WIDTH, height=self.HEIGHT,
                       distortion_model=CameraData.DistortionModel.RADIAL_TANGENTIAL,
                       K=np.eye(3), D=np.zeros(5), R=np.eye(3), P=np.zeros((3, 3)))

    # =========================================================================
    # ============================ get_ros_msg ================================
    # =========================================================================

    def test_get_ros_msg_rosbags_fields(self):
        """ ROSBAGS message has correct header, dims, distortion model, and matrices. """
        cam = self._make_camera(D=self.D)
        msg = cam.get_ros_msg(ROSMsgLibType.ROSBAGS)

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
        msg = cam.get_ros_msg(ROSMsgLibType.ROSBAGS)

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
            cam.get_ros_msg(ROSMsgLibType.NONE)


if __name__ == "__main__":
    unittest.main()

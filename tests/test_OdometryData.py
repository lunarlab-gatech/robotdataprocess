from copy import deepcopy
from decimal import Decimal
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.Data import ROSMsgLibType, TransformType
from robotdataprocess.data_types.OdometryData import OdometryData, PATH_SLICE_STEP
from robotdataprocess.data_types.PathData import PathData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag1 import Writer as Writer1
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
import tempfile
from test_utils import safe_urlretrieve
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestOdometryData(unittest.TestCase):

    def test_from_csv(self):
        """ 
        Test we can load data from csv files, with or without headers 
        and by specifying which columns have which data.
        """

        # ===== Test with no Header & extra data in file =====
        # Load the Odometry Data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_csv', 'vins_result_no_loop.csv').absolute()
        odom_data = OdometryData.from_csv(file_path, "odom", "base_link", CoordinateFrame.FLU, False, None)

        # Make sure it matches what we expect
        np.testing.assert_equal(float(odom_data.timestamps[0]), 7.7000000000)
        np.testing.assert_array_equal(odom_data.positions[0].astype(np.float128), [-0.0038540630,-0.0048488862,1.1692433748])
        np.testing.assert_array_equal(odom_data.orientations[0].astype(np.float128), [0.9975824644,-0.0002800578,0.0000495580,-0.0694920556])

        np.testing.assert_equal(float(odom_data.timestamps[54]), 10.4000000000)
        np.testing.assert_array_equal(odom_data.positions[54].astype(np.float128), [-0.0159722930,-1.8196936490,1.4511139975])
        np.testing.assert_array_equal(odom_data.orientations[54].astype(np.float128), [0.9953149851,-0.0001806001,0.0002772285, 0.0966849053])
        
        # ===== Test with header and no extra data =====
        # Load the Odometery Data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_csv', 'odomGT.csv').absolute()
        odom_data = OdometryData.from_csv(file_path, "odom", "base_link", CoordinateFrame.FLU, True, None)

        # Make sure it matches what we expect
        np.testing.assert_equal(float(odom_data.timestamps[0]), 0.050000)
        np.testing.assert_array_equal(odom_data.positions[0].astype(np.float128), [-0.001950,-0.000122,-1.445321])
        np.testing.assert_array_almost_equal(odom_data.orientations[0].astype(np.float128), [-0.001957000162432977,-4.400000365204445e-05,0.9999980830008444,4.700000390104748e-05], 16)

        np.testing.assert_equal(float(odom_data.timestamps[688]), 34.450000)
        np.testing.assert_array_equal(odom_data.positions[688].astype(np.float128), [-3.896535,-1.679678,-1.445265])
        np.testing.assert_array_almost_equal(odom_data.orientations[688].astype(np.float128), [0.0034349994640731434,0.00016199997472484692,-0.9997928440128326, 0.020060996870093543], 16)

        # ===== Test with header and a filter =====
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_csv', 'vertex_poses_velocities_biases.csv').absolute()
        odom_data = OdometryData.from_csv(file_path, "odom", "base_link", CoordinateFrame.NED, True, [0,3,4,5,6,7,8,9], filter=(' mission-id', ' 38a88adc194a7f180900000000000000'), ts_in_ns=True)

        # Make sure it matches what we expect
        np.testing.assert_equal(float(odom_data.timestamps[0]), 589.1)
        np.testing.assert_array_equal(odom_data.positions[0].astype(np.float128), [-68521.3784775933, -96139.6097434788, 3995.39987267099])
        np.testing.assert_array_equal(odom_data.orientations[0].astype(np.float128), [-0.68937326290076, -0.344671762209619, -0.636848081697329, 0.0197585822190744])

        np.testing.assert_equal(float(odom_data.timestamps[10]), 590.1)
        np.testing.assert_array_equal(odom_data.positions[10].astype(np.float128), [-68760.22833236, -97018.3443167711, 4776.87739072918])
        np.testing.assert_array_equal(odom_data.orientations[10].astype(np.float128), [0.586834741837591, 0.199208236471664, -0.784542763433785, 0.0208258646384441])

    def test_from_txt_AND_get_ros_msg_AND_from_ros2_bag(self):
        """
        Test that we can load Odometry data from a txt file 
        and save it into a ROS2 bag.
        """

        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.FLU, False)
        bag_path = Path(Path('.'), 'tests', 'test_bags', 'test_from_txt_file', 'odom_bag').absolute()
        if os.path.isdir(bag_path):
            if os.path.exists(bag_path / 'odom_bag.db3'):
                os.remove(bag_path / 'odom_bag.db3')
            if os.path.exists(bag_path / 'metadata.yaml'):
                os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)

        # Save it into a ROS2 bag
        Ros2BagWrapper.write_data_to_ros2_bag(bag_path, [odom_data, odom_data], ['/odom', '/odom/path'], ["Odometry", "Path"], None)

        # Load the data back again
        ros_data = OdometryData.from_ros2_bag(bag_path, '/odom', CoordinateFrame.FLU)

        # Make sure this data matches what we expect
        np.testing.assert_equal(float(ros_data.timestamps[32]), 690.100000)
        np.testing.assert_array_equal(ros_data.positions[32].astype(np.float128), [-66.153381, -76.155663, 1.445448])
        np.testing.assert_array_equal(ros_data.orientations[32].astype(np.float128), [0.001246, -0.000566, 0.916554, 0.399908])
        np.testing.assert_equal(ros_data.frame_id, '/Husky1')
        np.testing.assert_equal(ros_data.child_frame_id, '/Husky1/base_link')
        np.testing.assert_equal(ros_data.frame, CoordinateFrame.FLU)

        # Make sure the Odometry and Path options match in their data. 
        path_data = PathData.from_ros2_bag(bag_path, '/odom/path', CoordinateFrame.FLU)
        np.testing.assert_equal(math.ceil(ros_data.len() / PATH_SLICE_STEP), path_data.len())
        np.testing.assert_equal(ros_data.frame_id, path_data.frame_id)
        np.testing.assert_equal(ros_data.timestamps[PATH_SLICE_STEP], path_data.timestamps[1])
        np.testing.assert_array_equal(ros_data.positions[PATH_SLICE_STEP], path_data.positions[1])
        np.testing.assert_array_equal(ros_data.orientations[PATH_SLICE_STEP], path_data.orientations[1])

        # Check TFMessage via ROSBAGS
        tf_msg = odom_data.get_ros_msg(ROSMsgLibType.ROSBAGS, 32, "TFMessage")
        self.assertEqual(len(tf_msg.transforms), 1)
        ts = tf_msg.transforms[0]
        self.assertEqual(ts.header.frame_id, '/Husky1')
        self.assertEqual(ts.child_frame_id, '/Husky1/base_link')
        np.testing.assert_almost_equal(float(ts.transform.translation.x), float(odom_data.positions[32][0]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.translation.y), float(odom_data.positions[32][1]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.translation.z), float(odom_data.positions[32][2]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.rotation.x), float(odom_data.orientations[32][0]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.rotation.y), float(odom_data.orientations[32][1]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.rotation.z), float(odom_data.orientations[32][2]), decimal=6)
        np.testing.assert_almost_equal(float(ts.transform.rotation.w), float(odom_data.orientations[32][3]), decimal=6)

        # Check that TFMessage msg_type string is returned by get_ros_msg_type
        tf_msg_type = OdometryData.get_ros_msg_type(ROSMsgLibType.ROSBAGS, "TFMessage")
        self.assertEqual(tf_msg_type, 'tf2_msgs/msg/TFMessage')

        # Check that the new arguments work properly
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom_openvins.txt').absolute()
        odom_data_test = OdometryData.from_txt(file_path, 'frame', 'child_frame', CoordinateFrame.FLU, True, [0, 5, 6, 7, 4, 1, 2, 3])
        np.testing.assert_equal(odom_data_test.len(), 3)
        np.testing.assert_equal(odom_data_test.frame_id, 'frame')
        np.testing.assert_equal(odom_data_test.child_frame_id, 'child_frame')
        np.testing.assert_array_equal(odom_data_test.timestamps.astype(np.float64), np.array([1403715278.86375, 1403715278.91222, 1403715278.96200], dtype=np.float64))
        np.testing.assert_array_equal(odom_data_test.positions[1].astype(np.float64), [0.047673, 0.008781, 0.078044])
        np.testing.assert_array_equal(odom_data_test.orientations[2].astype(np.float64), [0.810142, -0.007347, 0.586184, 0.001775])

    def test_to_coordinate_frame(self):
        """
        Makes sure that the conversion from NED to FLU functions properly via to_coordinate_frame.
        """

        def compare_with_expected(odom_data: OdometryData):
            np.testing.assert_equal(float(odom_data.timestamps[32]), 690.100000)
            np.testing.assert_array_equal(odom_data.positions[32].astype(np.float128), [-66.153381, 76.155663, -1.445448])
            np.testing.assert_array_almost_equal(np.abs(odom_data.orientations[32].astype(np.float128)), np.abs([-0.0012460003013751132, -0.0005660001369007335, 0.9165542216906626, -0.3999080967273826]), 8)
            np.testing.assert_equal(odom_data.frame_id, '/Husky1')
            np.testing.assert_equal(odom_data.child_frame_id, '/Husky1/base_link')
            np.testing.assert_equal(odom_data.frame, CoordinateFrame.FLU)
            # Verify Decimal dtype is preserved after frame conversion
            self.assertIsInstance(odom_data.positions[0][0], Decimal)
            self.assertIsInstance(odom_data.orientations[0][0], Decimal)

        # ===  Test NED to FLU (default CHANGE_OF_BASIS) ===
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        odom_data.to_coordinate_frame(CoordinateFrame.FLU)
        compare_with_expected(odom_data)

        # === Test FLU to FLU (no-op) ===
        odom_data.to_coordinate_frame(CoordinateFrame.FLU)
        compare_with_expected(odom_data)

        # === Test Unsupported conversion throws error ===
        odom_data.frame = CoordinateFrame.ENU
        with np.testing.assert_raises(NotImplementedError):
            odom_data.to_coordinate_frame(CoordinateFrame.FLU)

        # === Test ROTATION vs CHANGE_OF_BASIS orientations ===
        odom_rotation = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)
        odom_cob = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)
        original_ori = odom_rotation.orientations.copy()

        odom_rotation.to_coordinate_frame(CoordinateFrame.FLU, TransformType.ROTATION)
        odom_cob.to_coordinate_frame(CoordinateFrame.FLU, TransformType.CHANGE_OF_BASIS)

        # Positions should be identical (both apply R * p)
        np.testing.assert_array_equal(odom_rotation.positions, odom_cob.positions)

        # Decimal dtype must be preserved after both transform types
        self.assertIsInstance(odom_rotation.positions[0][0], Decimal)
        self.assertIsInstance(odom_rotation.orientations[0][0], Decimal)
        self.assertIsInstance(odom_cob.positions[0][0], Decimal)
        self.assertIsInstance(odom_cob.orientations[0][0], Decimal)

        # Orientations should differ between ROTATION and CHANGE_OF_BASIS
        self.assertFalse(np.allclose(
            odom_rotation.orientations.astype(np.float64),
            odom_cob.orientations.astype(np.float64)
        ))

        # Verify orientations at specific indices against the mathematical definitions
        R_frame = R.from_euler('x', 180, degrees=True)
        for idx in [0, 32, 100]:
            q_orig = R.from_quat(original_ori[idx].astype(np.float64))

            # ROTATION: q_new = R_frame * q_original
            expected_rotation = (R_frame * q_orig).as_quat()
            np.testing.assert_array_almost_equal(
                odom_rotation.orientations[idx].astype(np.float64),
                expected_rotation, decimal=8
            )

            # CHANGE_OF_BASIS: q_new = R_frame * q_original * R_frame^{-1}
            expected_cob = (R_frame * q_orig * R_frame.inv()).as_quat()
            np.testing.assert_array_almost_equal(
                odom_cob.orientations[idx].astype(np.float64),
                expected_cob, decimal=8
            )

    def test_shift_to_start_at_identity(self):
        """
        Tests that we can properly shift a sequence of odometry data to start at the origin.
        """

        # Load the Odometry data and convert into the ROS frame
        file_path = Path(Path('.'), 'tests', 'test_outputs', 'test_from_txt_file', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)
        odom_data.to_coordinate_frame(CoordinateFrame.FLU)

        # Shift it so that it starts at the origin
        odom_data.shift_to_start_at_identity()

        # Make sure the data matches what we expect
        np.testing.assert_equal(float(odom_data.timestamps[13801]), 690.100000)
        np.testing.assert_array_almost_equal(odom_data.positions[13801].astype(np.float128), [66.16544698000006, -76.15057619688778, 0.25349471896643494], 2)
        np.testing.assert_array_almost_equal(odom_data.orientations[13801].astype(np.float128), [-0.0013123360311483368, -0.0005744812796045746, 0.3999401764357198, 0.9165401262454177], 8)
        np.testing.assert_equal(odom_data.frame_id, '/Husky1')
        np.testing.assert_equal(odom_data.child_frame_id, '/Husky1/base_link')
        np.testing.assert_equal(odom_data.frame, CoordinateFrame.FLU)

    def test_crop_data(self):
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_crop_data', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        # Test cropping out some data
        odom_data_cropped = deepcopy(odom_data)
        odom_data_cropped.crop_data(Decimal('0.45'), Decimal('2.95')) 
        np.testing.assert_array_equal(odom_data_cropped.timestamps, odom_data.timestamps[8:59])
        np.testing.assert_array_equal(odom_data_cropped.positions, odom_data.positions[8:59])
        np.testing.assert_array_equal(odom_data_cropped.orientations, odom_data.orientations[8:59])

    def test_ori_apply_rotation(self):
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_ori_apply_rotation', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        # Ensure the rotation functions properly
        odom_data_rotated = deepcopy(odom_data)
        rotation = R.from_quat([0.7071068, 0, 0, 0.7071068])
        odom_data_rotated._ori_apply_rotation_left_side(rotation)
        np.testing.assert_array_almost_equal(odom_data_rotated.orientations[10].astype(np.float128), np.array([-0.00136472,  0.70713652, -0.7070743, 0.00141704]), 8)

    def test_ori_apply_rotation_right_side(self):
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_ori_apply_rotation', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        # Ensure the rotation functions properly and matches R.from_quat(q) * R_i
        odom_data_rotated = deepcopy(odom_data)
        rotation = R.from_quat([0.7071068, 0, 0, 0.7071068])
        odom_data_rotated._ori_apply_rotation_right_side(rotation)
        expected = (R.from_quat(odom_data.orientations[10].astype(np.float128)) * rotation).as_quat()
        np.testing.assert_array_almost_equal(odom_data_rotated.orientations[10].astype(np.float128), expected, 8)

    def test_ori_apply_rotation_right_side_multiple_orientations(self):
        # Use several distinct, non-trivial orientations to make sure each row is transformed independently
        orientations = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.7071068, 0.0, 0.0, 0.7071068],
            [0.0, 0.7071068, 0.0, 0.7071068],
            [0.2705981, 0.2705981, 0.6532815, 0.6532815],
        ])
        odom_data = OdometryData("temp", "child", [0, 1, 2, 3], np.zeros((4, 3)), orientations, CoordinateFrame.FLU)

        rotation = R.from_euler('xyz', [15, -30, 60], degrees=True)
        odom_data._ori_apply_rotation_right_side(rotation)

        for i in range(len(orientations)):
            expected = (R.from_quat(orientations[i]) * rotation).as_quat()
            np.testing.assert_array_almost_equal(odom_data.orientations[i].astype(np.float128), expected, 6)

    def test_ori_apply_rotation_right_side_identity_is_noop(self):
        # Rotating by the identity should leave every orientation unchanged
        orientations = np.array([
            [0.2705981, 0.2705981, 0.6532815, 0.6532815],
            [-0.5, 0.5, -0.5, 0.5],
        ])
        odom_data = OdometryData("temp", "child", [0, 1], np.zeros((2, 3)), orientations, CoordinateFrame.FLU)

        odom_data._ori_apply_rotation_right_side(R.identity())

        np.testing.assert_array_almost_equal(odom_data.orientations.astype(np.float128), orientations, 6)

    def test_ori_apply_rotation_right_side_order_matters(self):
        # Right-side and left-side application should differ for non-commuting rotations
        orientation = np.array([[0.2705981, 0.2705981, 0.6532815, 0.6532815]])
        rotation = R.from_euler('xyz', [10, 20, 30], degrees=True)

        odom_right = OdometryData("temp", "child", [0], np.zeros((1, 3)), orientation.copy(), CoordinateFrame.FLU)
        odom_right._ori_apply_rotation_right_side(rotation)

        odom_left = OdometryData("temp", "child", [0], np.zeros((1, 3)), orientation.copy(), CoordinateFrame.FLU)
        odom_left._ori_apply_rotation_left_side(rotation)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                odom_right.orientations[0].astype(np.float128), odom_left.orientations[0].astype(np.float128), 6)

        expected_right = (R.from_quat(orientation[0]) * rotation).as_quat()
        np.testing.assert_array_almost_equal(odom_right.orientations[0].astype(np.float128), expected_right, 8)

    def test_ori_apply_rotation_right_side_preserves_unit_norm(self):
        orientations = np.array([
            [0.2705981, 0.2705981, 0.6532815, 0.6532815],
            [0.0, 0.0, 0.0, 1.0],
            [-0.5, 0.5, -0.5, 0.5],
        ])
        odom_data = OdometryData("temp", "child", [0, 1, 2], np.zeros((3, 3)), orientations, CoordinateFrame.FLU)

        rotation = R.from_euler('xyz', [42, -17, 8], degrees=True)
        odom_data._ori_apply_rotation_right_side(rotation)

        norms = np.linalg.norm(odom_data.orientations.astype(np.float128), axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), 8)

    def test_ori_apply_rotation_right_side_leaves_positions_and_timestamps_unchanged(self):
        timestamps = [Decimal("0.5"), Decimal("1.5")]
        positions = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
        orientations = np.array([[0.0, 0.0, 0.0, 1.0], [0.7071068, 0.0, 0.0, 0.7071068]])
        odom_data = OdometryData("temp", "child", timestamps, positions, orientations, CoordinateFrame.FLU)

        odom_data._ori_apply_rotation_right_side(R.from_euler('xyz', [5, 10, 15], degrees=True))

        np.testing.assert_array_equal(odom_data.positions.astype(float), positions)
        np.testing.assert_array_equal([float(ts) for ts in odom_data.timestamps], [0.5, 1.5])

    def test_apply_transformation(self):
        # Load the Odometry data
        odom_data = OdometryData("temp", "child", [0], np.array([[1, 2, 3]]), np.array([[-0.7071068, 0, 0, 0.7071068]]), CoordinateFrame.FLU)

        # Apply the transformation and make sure it worked
        H = np.array([[0.0, -1.0,  0.0,  1.0],
                      [1.0,  0.0,  0.0,  0.0],
                      [0.0,  0.0,  1.0,  0.0],
                      [0.0,  0.0,  0.0,  1.0]])
        odom_data.apply_transformation_left_side(H)

        np.testing.assert_array_equal(odom_data.positions[0].astype(float), np.array([-1, 1, 3]))
        np.testing.assert_array_almost_equal(odom_data.orientations[0].astype(float), np.array([0.5, 0.5, -0.5, -0.5]), decimal=6)

        # Test with a more complicated transformation
        odom_data = OdometryData("temp", "child", [0], np.array([[1, 2, 3]]), np.array([[-0.7071068, 0, 0, 0.7071068]]), CoordinateFrame.FLU)

        # Apply the transformation and make sure it worked
        odom_data.apply_transformation_right_side(H)

        np.testing.assert_array_equal(odom_data.positions[0].astype(float), np.array([2, 2, 3]))
        np.testing.assert_array_almost_equal(odom_data.orientations[0].astype(float), np.array([ 0.5, -0.5, -0.5, -0.5 ]), decimal=6)

    def test_to_csv(self):
        """ Test that we can extract odometry from a ROS2 bag and save to CSV correctly. """

        # Setup paths and download test bag if needed
        path_hercules_bag = Path(Path('.'), 'tests', 'test_bags', 'hercules_test_bag_pruned_3_FINAL').absolute()
        path_hercules_bag_db3 = path_hercules_bag / Path("hercules_test_bag_pruned_3_FINAL.db3")
        path_hercules_bag_yaml = path_hercules_bag / Path("metadata.yaml")

        if not os.path.isfile(path_hercules_bag_db3):
            safe_urlretrieve("https://www.dropbox.com/scl/fi/0ydrblh1uai1lhbrrk6c6/hercules_test_bag_pruned_3_FINAL.db3?rlkey=n27tgr0vuxcyrsyafavlh0aw9&st=i5qixbjo&dl=1", path_hercules_bag_db3)
        if not os.path.isfile(path_hercules_bag_yaml):
            safe_urlretrieve("https://www.dropbox.com/scl/fi/2iu1djmhedy1j4qci53a4/metadata.yaml?rlkey=x0kb00pruubxtbaojht4ui5yl&st=9b41wq07&dl=1", path_hercules_bag_yaml)

        # Define output path
        output_file = Path(Path('.'), 'tests', 'temporary_files', 'test_OdometryData', 'test_to_csv', 'Husky2_odom.csv').absolute()
        topic = "/hercules_node/Husky2/ground_truth/odom_local"

        # Delete output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Create output folder if it doesn't exist
        output_folder = output_file.parent
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load odometry from ROS2 bag and save to CSV
        odom_data = OdometryData.from_ros2_bag(path_hercules_bag, topic, CoordinateFrame.NONE)
        odom_data.to_csv(output_file)

        # Load csv file and check values
        df = pd.read_csv(output_file)

        first_row = df.iloc[0].tolist()
        np.testing.assert_almost_equal(first_row[0], 1749131152.801460736, 14)
        np.testing.assert_almost_equal(first_row[1], -0.0000245371702476404607295989990234375, 14)
        np.testing.assert_almost_equal(first_row[2], -0.0000033797959986259229481220245361328125, 14)
        np.testing.assert_almost_equal(first_row[3], -1.44854152202606201171875, 14)
        np.testing.assert_almost_equal(first_row[4], 0.99998915195465087890625, 14)
        np.testing.assert_almost_equal(first_row[5], -0.00005158343992661684751586201171875, 14)
        np.testing.assert_almost_equal(first_row[6], 0.004659599624574184417724609375, 14)
        np.testing.assert_almost_equal(first_row[7], 1.05546661188782309181988239288330078125E-7, 14)

        random_row = df[df['timestamp'] == 1749131152.883519488].iloc[0]
        np.testing.assert_almost_equal(random_row['x'], -0.0000245371702476404607295989990234375, 14)
        np.testing.assert_almost_equal(random_row['y'], -0.0000033797959986259229481220245361328125, 14)
        np.testing.assert_almost_equal(random_row['z'], -1.44854152202606201171875, 14)
        np.testing.assert_almost_equal(random_row['qw'], 0.99998915195465087890625, 14)
        np.testing.assert_almost_equal(random_row['qx'], -0.000051583439926616847515106201171875, 14)
        np.testing.assert_almost_equal(random_row['qy'], 0.004659599624574184417724609375, 14)
        np.testing.assert_almost_equal(random_row['qz'], 1.05546661188782309181988239288330078125E-7, 14)


    def test_shift_position(self):
        """ Test that shift_position modifies positions correctly. """
        odom = OdometryData("world", "robot", [0, 1, 2],
                            np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0],
                                      [7.0, 8.0, 9.0]]),
                            np.array([[0, 0, 0, 1],
                                      [0, 0, 0, 1],
                                      [0, 0, 0, 1]]),
                            CoordinateFrame.FLU)
        odom.shift_position(10.0, -20.0, 5.0)
        np.testing.assert_array_almost_equal(
            odom.positions.astype(float),
            [[11.0, -18.0, 8.0],
             [14.0, -15.0, 11.0],
             [17.0, -12.0, 14.0]])

    def test_interpolate_to_hz(self):
        """ Test interpolation to a target frequency including SLERP for orientations. """
        # Use rotations about Z: 0 deg, 90 deg, 180 deg at t=0, 1, 2
        r0 = R.from_euler('z', 0, degrees=True).as_quat()
        r90 = R.from_euler('z', 90, degrees=True).as_quat()
        r180 = R.from_euler('z', 180, degrees=True).as_quat()

        odom = OdometryData("world", "robot",
                            np.array([0.0, 1.0, 2.0]),
                            np.array([[0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0],
                                      [2.0, 0.0, 0.0]]),
                            np.array([r0, r90, r180]),
                            CoordinateFrame.FLU)
        odom.interpolate_to_hz(2.0)

        # Check positions are linearly interpolated
        self.assertEqual(odom.len(), 5)
        np.testing.assert_array_almost_equal(
            odom.positions.astype(float)[:, 0],
            [0.0, 0.5, 1.0, 1.5, 2.0])

        # Check orientations are SLERPed correctly
        expected_quats = [R.from_euler('z', deg, degrees=True).as_quat()
                          for deg in [0, 45, 90, 135, 180]]
        for i, expected_quat in enumerate(expected_quats):
            np.testing.assert_array_almost_equal(
                odom.orientations[i].astype(float), expected_quat, decimal=10)

        # Check ValueError for non-positive hz
        with self.assertRaises(ValueError):
            odom2 = OdometryData("world", "robot", [0, 1],
                                 np.array([[0, 0, 0], [1, 0, 0]]),
                                 np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                                 CoordinateFrame.FLU)
            odom2.interpolate_to_hz(0)
        with self.assertRaises(ValueError):
            odom3 = OdometryData("world", "robot", [0, 1],
                                 np.array([[0, 0, 0], [1, 0, 0]]),
                                 np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                                 CoordinateFrame.FLU)
            odom3.interpolate_to_hz(-1.0)

    def test_to_csv_existing_file_raises(self):
        """ Test that to_csv raises ValueError when file already exists. """
        odom = OdometryData("world", "robot", [0],
                            np.array([[1.0, 2.0, 3.0]]),
                            np.array([[0, 0, 0, 1]]),
                            CoordinateFrame.FLU)
        existing_file = Path(__file__).absolute()
        with self.assertRaises(ValueError):
            odom.to_csv(existing_file)

    def test_from_tum(self):
        """
        Test that OdometryData.from_tum() correctly loads a TUM-format file.
        TUM format: timestamp x y z qx qy qz qw (space-separated, no header).
        Orientations must be stored internally as (qx, qy, qz, qw).
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tum_file = Path(d) / "tum.txt"
            # Three rows: ts x y z qx qy qz qw
            tum_file.write_text(
                "1.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0\n"
                "2.0 4.0 5.0 6.0 0.5 0.5 0.5 0.5\n"
                "3.0 7.0 8.0 9.0 0.1 0.2 0.3 0.9\n"
            )

            odom = OdometryData.from_tum(tum_file, "world", "robot", CoordinateFrame.FLU)

            # Check type and metadata
            self.assertIsInstance(odom, OdometryData)
            self.assertEqual(odom.frame_id, "world")
            self.assertEqual(odom.child_frame_id, "robot")
            self.assertEqual(odom.frame, CoordinateFrame.FLU)
            self.assertEqual(odom.len(), 3)

            # Check timestamps
            np.testing.assert_array_almost_equal(
                odom.timestamps.astype(float), [1.0, 2.0, 3.0])

            # Check positions
            np.testing.assert_array_almost_equal(
                odom.positions[0].astype(float), [1.0, 2.0, 3.0])
            np.testing.assert_array_almost_equal(
                odom.positions[1].astype(float), [4.0, 5.0, 6.0])
            np.testing.assert_array_almost_equal(
                odom.positions[2].astype(float), [7.0, 8.0, 9.0])

            # Check orientations stored as (qx, qy, qz, qw)
            np.testing.assert_array_almost_equal(
                odom.orientations[0].astype(float), [0.0, 0.0, 0.0, 1.0])
            np.testing.assert_array_almost_equal(
                odom.orientations[1].astype(float), [0.5, 0.5, 0.5, 0.5])
            np.testing.assert_array_almost_equal(
                odom.orientations[2].astype(float), [0.1, 0.2, 0.3, 0.9])

    def test_from_tum_column_order(self):
        """
        from_tum() must read qw from the last (8th) column, not the 5th.
        Verifies the TUM mapping ts x y z qx qy qz qw is applied correctly.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tum_file = Path(d) / "manual.txt"
            # ts=5.0  x=1  y=2  z=3  qx=0.1  qy=0.2  qz=0.3  qw=0.9
            tum_file.write_text("5.0 1.0 2.0 3.0 0.1 0.2 0.3 0.9\n")

            odom = OdometryData.from_tum(tum_file, "map", "base_link", CoordinateFrame.FLU)
            self.assertEqual(odom.len(), 1)
            self.assertAlmostEqual(float(odom.timestamps[0]), 5.0)
            np.testing.assert_array_almost_equal(
                odom.positions[0].astype(float), [1.0, 2.0, 3.0])
            # qw=0.9 must be last; qx=0.1, qy=0.2, qz=0.3
            np.testing.assert_array_almost_equal(
                odom.orientations[0].astype(float), [0.1, 0.2, 0.3, 0.9])

    def test_from_tum_round_trip(self):
        """
        Writing with to_tum() and reading back with OdometryData.from_tum()
        should reproduce the original data exactly.
        """
        import tempfile

        odom = OdometryData(
            "world", "robot",
            np.array([10.0, 20.0], dtype=object),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=object),
            np.array([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]], dtype=object),
            CoordinateFrame.FLU
        )

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "tum.txt"
            odom.to_tum(out)

            loaded = OdometryData.from_tum(out, "world", "robot", CoordinateFrame.FLU)

            self.assertIsInstance(loaded, OdometryData)
            self.assertEqual(loaded.frame_id, "world")
            self.assertEqual(loaded.child_frame_id, "robot")
            self.assertEqual(loaded.frame, CoordinateFrame.FLU)
            np.testing.assert_array_almost_equal(
                loaded.timestamps.astype(float), odom.timestamps.astype(float))
            np.testing.assert_array_almost_equal(
                loaded.positions.astype(float), odom.positions.astype(float))
            np.testing.assert_array_almost_equal(
                loaded.orientations.astype(float), odom.orientations.astype(float))

    def _write_ros1_odom_bag(self, bag_path: Path, topic: str, frame_id: str,
                              child_frame_id: str, timestamps_sec: list,
                              positions: np.ndarray, orientations: np.ndarray) -> None:
        """Write a ROS1 bag containing nav_msgs/msg/Odometry messages."""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        OdomMsg       = typestore.types['nav_msgs/msg/Odometry']
        Header        = typestore.types['std_msgs/msg/Header']
        Time          = typestore.types['builtin_interfaces/msg/Time']
        PoseWithCov   = typestore.types['geometry_msgs/msg/PoseWithCovariance']
        Pose          = typestore.types['geometry_msgs/msg/Pose']
        Point         = typestore.types['geometry_msgs/msg/Point']
        Quaternion    = typestore.types['geometry_msgs/msg/Quaternion']
        TwistWithCov  = typestore.types['geometry_msgs/msg/TwistWithCovariance']
        Twist         = typestore.types['geometry_msgs/msg/Twist']
        Vector3       = typestore.types['geometry_msgs/msg/Vector3']

        with Writer1(bag_path) as writer:
            conn = writer.add_connection(topic, OdomMsg.__msgtype__, typestore=typestore)
            for i, (ts, pos, ori) in enumerate(zip(timestamps_sec, positions, orientations)):
                ts_dec = Decimal(str(ts))
                sec  = int(ts_dec)
                nsec = int((ts_dec - Decimal(sec)) * Decimal('1e9'))
                ts_ns = sec * 10**9 + nsec
                msg = OdomMsg(
                    Header(seq=i, stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id),
                    child_frame_id=child_frame_id,
                    pose=PoseWithCov(
                        pose=Pose(
                            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                            orientation=Quaternion(x=float(ori[0]), y=float(ori[1]),
                                                   z=float(ori[2]), w=float(ori[3])),
                        ),
                        covariance=np.zeros(36),
                    ),
                    twist=TwistWithCov(
                        twist=Twist(
                            linear=Vector3(x=0.0, y=0.0, z=0.0),
                            angular=Vector3(x=0.0, y=0.0, z=0.0),
                        ),
                        covariance=np.zeros(36),
                    ),
                )
                writer.write(conn, ts_ns, typestore.serialize_ros1(msg, OdomMsg.__msgtype__))

    def test_from_ros1_bag(self):
        """Write a ROS1 bag with known Odometry messages and verify from_ros1_bag round-trips."""
        frame_id       = 'odom'
        child_frame_id = 'base_link'
        topic          = '/odom'
        timestamps_sec = [1.0, 2.0, 3.0]
        positions      = np.array([[1.1, 2.2, 3.3],
                                   [4.4, 5.5, 6.6],
                                   [7.7, 8.8, 9.9]])
        # orientations as (qx, qy, qz, qw)
        orientations   = np.array([[0.0, 0.0, 0.0,        1.0       ],
                                   [0.5, 0.5, 0.5,        0.5       ],
                                   [0.0, 0.0, 0.70710678, 0.70710678]])

        with tempfile.TemporaryDirectory() as tmpdir:
            bag_path = Path(tmpdir) / 'odom.bag'
            self._write_ros1_odom_bag(bag_path, topic, frame_id, child_frame_id,
                                      timestamps_sec, positions, orientations)

            data = OdometryData.from_ros1_bag(bag_path, topic, CoordinateFrame.FLU)

            # --- metadata ---
            self.assertEqual(data.frame_id, frame_id)
            self.assertEqual(data.child_frame_id, child_frame_id)
            self.assertEqual(data.frame, CoordinateFrame.FLU)
            self.assertEqual(data.len(), 3)

            # --- timestamps ---
            np.testing.assert_array_almost_equal(
                data.timestamps.astype(np.float64), timestamps_sec, decimal=6)

            # --- positions ---
            np.testing.assert_array_almost_equal(
                data.positions.astype(np.float64), positions, decimal=6)

            # --- orientations ---
            np.testing.assert_array_almost_equal(
                data.orientations.astype(np.float64), orientations, decimal=6)


    def _make_odom_data(self):
        return OdometryData(
            frame_id="world",
            child_frame_id="base_link",
            timestamps=np.array([1.0, 2.0, 3.0], dtype=object),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=object),
            orientations=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=object),
            frame=CoordinateFrame.FLU,
        )

    def test_eq(self):
        from copy import deepcopy
        from decimal import Decimal

        o1 = self._make_odom_data()
        o2 = deepcopy(o1)
        self.assertEqual(o1, o2)

        # child_frame_id differs
        o = deepcopy(o1); o.child_frame_id = "other_link"
        self.assertNotEqual(o1, o)

        # frame_id differs (inherited from Data)
        o = deepcopy(o1); o.frame_id = "other"
        self.assertNotEqual(o1, o)

        # timestamps differ (inherited from SequentialData)
        o = deepcopy(o1); o.timestamps[0] = Decimal("9.0")
        self.assertNotEqual(o1, o)

        # positions differ (inherited from PathData)
        o = deepcopy(o1); o.positions[0, 0] = Decimal("99.0")
        self.assertNotEqual(o1, o)

        # orientations differ (inherited from PathData)
        o = deepcopy(o1); o.orientations[0, 0] = Decimal("0.5")
        self.assertNotEqual(o1, o)

        # frame differs (inherited from PathData)
        o = deepcopy(o1); o.frame = CoordinateFrame.NED
        self.assertNotEqual(o1, o)

        # OdometryData vs PathData is not equal even when common fields match
        path = PathData(
            frame_id=o1.frame_id,
            timestamps=o1.timestamps,
            positions=o1.positions,
            orientations=o1.orientations,
            frame=o1.frame,
        )
        self.assertNotEqual(o1, path)

        # poses / poses_rclpy caches do not affect equality
        o = deepcopy(o1)
        o.poses = ["fake"]
        o.poses_rclpy = ["fake"]
        self.assertEqual(o1, o)


if __name__ == "__main__":
    unittest.main()
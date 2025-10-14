from copy import deepcopy
from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.OdometryData import OdometryData
from robotdataprocess.data_types.PathData import PathData
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R
import unittest

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

    def test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag(self):
        """
        Test that we can load Odometry data from a txt file 
        and save it into a ROS2 bag.
        """

        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.FLU)
        bag_path = Path(Path('.'), 'tests', 'test_bags', 'test_from_txt_file', 'odom_bag').absolute()
        if os.path.isdir(bag_path):
            os.remove(bag_path / 'odom_bag.db3')
            os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)

        # Save it into a ROS2 bag
        Ros2BagWrapper.write_data_to_rosbag(bag_path, [odom_data, odom_data], ['/odom', '/odom/path'], ["Odometry", "Path"], None)

        # Load the data back again
        ros_data = OdometryData.from_ros2_bag(bag_path, '/odom')

        # Make sure this data matches what we expect
        np.testing.assert_equal(float(ros_data.timestamps[32]), 690.100000)
        np.testing.assert_array_equal(ros_data.positions[32].astype(np.float128), [-66.153381, -76.155663, 1.445448])
        np.testing.assert_array_equal(ros_data.orientations[32].astype(np.float128), [0.001246, -0.000566, 0.916554, 0.399908])
        np.testing.assert_equal(ros_data.frame_id, '/Husky1')
        np.testing.assert_equal(ros_data.child_frame_id, '/Husky1/base_link')
        np.testing.assert_equal(ros_data.frame, CoordinateFrame.FLU)

        # Make sure the Odometry and Path options match in their data. 
        path_data = PathData.from_ros2_bag(bag_path, '/odom/path')
        np.testing.assert_equal(ros_data.len(), path_data.len() * 10)
        np.testing.assert_equal(ros_data.frame_id, path_data.frame_id)
        np.testing.assert_equal(ros_data.timestamps[30], path_data.timestamps[3])
        np.testing.assert_array_equal(ros_data.positions[30], path_data.positions[3])
        np.testing.assert_array_equal(ros_data.orientations[30], path_data.orientations[3])

    def test_to_FLU_frame(self):
        """ 
        Makes sure that the conversion from NED to ROS functions properly.
        """

        def compare_with_expected(odom_data: OdometryData):
            np.testing.assert_equal(float(odom_data.timestamps[32]), 690.100000)
            np.testing.assert_array_equal(odom_data.positions[32].astype(np.float128), [-66.153381, 76.155663, -1.445448])
            np.testing.assert_array_almost_equal(odom_data.orientations[32].astype(np.float128), [0.0012460003013751132, 0.0005660001369007335, -0.9165542216906626, 0.3999080967273826], 8)
            np.testing.assert_equal(odom_data.frame_id, '/Husky1')
            np.testing.assert_equal(odom_data.child_frame_id, '/Husky1/base_link')
            np.testing.assert_equal(odom_data.frame, CoordinateFrame.FLU)

        # ===  Test NED to FLU ===
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED)

        # Converts it into the FLU coordinate system
        odom_data.to_FLU_frame()
        compare_with_expected(odom_data)

        # === Test FLU to FLU ===
        # Try to convert again, should do nothing.
        odom_data.to_FLU_frame()
        compare_with_expected(odom_data)

        # === Test Unsupported formats throw error ===
        odom_data.frame = CoordinateFrame.ENU
        with np.testing.assert_raises(RuntimeError):
            odom_data.to_FLU_frame()

    def test_shift_to_start_at_identity(self):
        """
        Tests that we can properly shift a sequence of odometry data to start at the origin.
        """

        # Load the Odometry data and convert into the ROS frame
        file_path = Path(Path('.'), 'tests', 'test_outputs', 'test_from_txt_file', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED)
        odom_data.to_FLU_frame()

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
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED)

        # Test cropping out some data
        odom_data_cropped = deepcopy(odom_data)
        odom_data_cropped.crop_data(Decimal('0.45'), Decimal('2.95')) 
        np.testing.assert_array_equal(odom_data_cropped.timestamps, odom_data.timestamps[8:59])
        np.testing.assert_array_equal(odom_data_cropped.positions, odom_data.positions[8:59])
        np.testing.assert_array_equal(odom_data_cropped.orientations, odom_data.orientations[8:59])

    def test_ori_apply_rotation(self):
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_ori_apply_rotation', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED)

        # Ensure the rotation functions properly
        odom_data_rotated = deepcopy(odom_data)
        rotation = R.from_quat([0.7071068, 0, 0, 0.7071068])
        odom_data_rotated._ori_apply_rotation(rotation)
        np.testing.assert_array_almost_equal(odom_data_rotated.orientations[10], np.array([-0.00136472,  0.70713652, -0.7070743, 0.00141704]), 8)

if __name__ == "__main__":
    unittest.main()
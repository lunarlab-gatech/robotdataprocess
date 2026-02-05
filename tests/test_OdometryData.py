from copy import deepcopy
from decimal import Decimal
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.OdometryData import OdometryData, PATH_SLICE_STEP
from robotdataprocess.data_types.PathData import PathData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R
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

    def test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag(self):
        """
        Test that we can load Odometry data from a txt file 
        and save it into a ROS2 bag.
        """

        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.FLU, False)
        bag_path = Path(Path('.'), 'tests', 'test_bags', 'test_from_txt_file', 'odom_bag').absolute()
        if os.path.isdir(bag_path):
            if os.path.exists(bag_path / 'odom_bag.db3'):
                os.remove(bag_path / 'odom_bag.db3')
            if os.path.exists(bag_path / 'metadata.yaml'):
                os.remove(bag_path / 'metadata.yaml')
            os.rmdir(bag_path)

        # Save it into a ROS2 bag
        Ros2BagWrapper.write_data_to_rosbag(bag_path, [odom_data, odom_data], ['/odom', '/odom/path'], ["Odometry", "Path"], None)

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

        # Check that the new arguments work properly
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom_openvins.txt').absolute()
        odom_data_test = OdometryData.from_txt_file(file_path, 'frame', 'child_frame', CoordinateFrame.FLU, True, [0, 5, 6, 7, 4, 1, 2, 3])
        np.testing.assert_equal(odom_data_test.len(), 3)
        np.testing.assert_equal(odom_data_test.frame_id, 'frame')
        np.testing.assert_equal(odom_data_test.child_frame_id, 'child_frame')
        np.testing.assert_array_equal(odom_data_test.timestamps.astype(np.float64), np.array([1403715278.86375, 1403715278.91222, 1403715278.96200], dtype=np.float64))
        np.testing.assert_array_equal(odom_data_test.positions[1].astype(np.float64), [0.047673, 0.008781, 0.078044])
        np.testing.assert_array_equal(odom_data_test.orientations[2].astype(np.float64), [0.810142, -0.007347, 0.586184, 0.001775])

    def test_to_FLU_frame(self):
        """ 
        Makes sure that the conversion from NED to ROS functions properly.
        """

        def compare_with_expected(odom_data: OdometryData):
            np.testing.assert_equal(float(odom_data.timestamps[32]), 690.100000)
            np.testing.assert_array_equal(odom_data.positions[32].astype(np.float128), [-66.153381, 76.155663, -1.445448])
            np.testing.assert_array_almost_equal(np.abs(odom_data.orientations[32].astype(np.float128)), np.abs([-0.0012460003013751132, -0.0005660001369007335, 0.9165542216906626, -0.3999080967273826]), 8)
            np.testing.assert_equal(odom_data.frame_id, '/Husky1')
            np.testing.assert_equal(odom_data.child_frame_id, '/Husky1/base_link')
            np.testing.assert_equal(odom_data.frame, CoordinateFrame.FLU)

        # ===  Test NED to FLU ===
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_from_txt_file_AND_get_ros_msg_AND_from_ros2_bag', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

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
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)
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
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        # Test cropping out some data
        odom_data_cropped = deepcopy(odom_data)
        odom_data_cropped.crop_data(Decimal('0.45'), Decimal('2.95')) 
        np.testing.assert_array_equal(odom_data_cropped.timestamps, odom_data.timestamps[8:59])
        np.testing.assert_array_equal(odom_data_cropped.positions, odom_data.positions[8:59])
        np.testing.assert_array_equal(odom_data_cropped.orientations, odom_data.orientations[8:59])

    def test_ori_apply_rotation(self):
        # Load the Odometry data
        file_path = Path(Path('.'), 'tests', 'files', 'test_OdometryData', 'test_ori_apply_rotation', 'odom.txt').absolute()
        odom_data = OdometryData.from_txt_file(file_path, '/Husky1', '/Husky1/base_link', CoordinateFrame.NED, False)

        # Ensure the rotation functions properly
        odom_data_rotated = deepcopy(odom_data)
        rotation = R.from_quat([0.7071068, 0, 0, 0.7071068])
        odom_data_rotated._ori_apply_rotation(rotation)
        np.testing.assert_array_almost_equal(odom_data_rotated.orientations[10], np.array([-0.00136472,  0.70713652, -0.7070743, 0.00141704]), 8)

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


if __name__ == "__main__":
    unittest.main()
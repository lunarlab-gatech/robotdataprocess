from decimal import Decimal
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame, LiDARData
import unittest

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLiDARData(unittest.TestCase):

    def test_from_npy_files(self):
        """ Ensure we load a point cloud from a folder with npy files correctly. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)

        # Check that the values are what we expect
        np.testing.assert_array_equal(lidar_data.timestamps.astype(float), [0.1, 0.6, 1.1])
        np.testing.assert_equal(lidar_data.frame, CoordinateFrame.NED)
        np.testing.assert_equal(lidar_data.frame_id, "robot")
        np.testing.assert_array_equal(lidar_data.point_clouds[0,3], [np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(lidar_data.point_clouds[0,35], [26.67838478088379, 0.3280501961708069, -9.796014785766602])
        np.testing.assert_array_equal(lidar_data.point_clouds[2,-1], [5.671111583709717, 9.91832280305971e-7, 2.6444828510284424])
        np.testing.assert_equal(lidar_data.point_clouds.shape[0], 3)

    def test_to_FLU_frame(self):
        """ Ensure our transformations are correct """

        point_cloud_simple = np.array([[[ 0,  1, 2],
                                        [-1, -3, 4]]])
        lidar_data = LiDARData("robot", [0], point_cloud_simple, None, CoordinateFrame.NED)
        lidar_data.to_FLU_frame()

        # Assert that the transformation completed successfully
        expected_pc = np.array([[[ 0, -1, -2], [-1,  3, -4]]])
        np.testing.assert_array_equal(lidar_data.point_clouds, expected_pc)

    def test_visualize(self):
        """ Just ensure that this code doesn't crash. """

        folder_path = Path(Path('.'), 'tests', 'files', 'test_LiDARData', 'test_from_npy_files').absolute()
        print(folder_path)
        lidar_data = LiDARData.from_npy_files(folder_path, "robot", CoordinateFrame.NED)
        lidar_data.visualize(testing=True)

if __name__ == "__main__":
    unittest.main()
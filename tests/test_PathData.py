import numpy
from pathlib import Path
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from robotdataprocess.data_types.PathData import PathData
import unittest

class TestPathData(unittest.TestCase):

    def test_calculate_trajectory_errors(self):
        """ Verify via regression test on a couple calculated metrics. """

        # Load the poseData files
        file_path = Path(__file__).parent / 'files' / 'test_PathData' / 'test_calculate_trajectory_errors'
        gt_data = OdometryData.from_csv(file_path / 'poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        est_data = OdometryData.from_csv(file_path / 'poseEst.csv', "world", "robot", CoordinateFrame.FLU, True, None)

        # Calculate all metrics
        results_dict: dict = PathData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1)
        
        # Make sure the values match what we expect
        numpy.testing.assert_almost_equal(results_dict['APE']['translation_part']['rmse'], 0.43900241699624326, 12)
        numpy.testing.assert_almost_equal(results_dict['APE']['translation_part']['max'], 0.5769000332405032, 12)
        numpy.testing.assert_almost_equal(results_dict['APE']['rotation_angle_deg']['mean'], 35.1468632257006, 12)

        # TODO: Write test cases to verify that RPE metrics are good.


if __name__ == "__main__":
    unittest.main()
from decimal import Decimal
import re
import getpass
import numpy as np
from pathlib import Path
from robotdataprocess import TransformationData
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["12_07_acl_jackal2"]
    dataset_version = "campus_tunnels_1207_compressed"
    dataset_number = re.search(r'\d+', dataset_version).group()  # e.g. "1207"
    robot_name_text = re.sub(r'(\d+_)+', '', robot_names[0])

    for robot_name in robot_names:
        print("\n=== Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset'

        # Load the data
        est_data = OdometryData.from_csv(dataset_folder / 'results' / dataset_number / robot_name_text / 'traj_vio.csv', 
                                        "world", "robot", CoordinateFrame.NONE, True, None, ts_in_ns=True)
        gt_data = OdometryData.from_csv(dataset_folder / 'data' / 'ground_truth' / dataset_number / (robot_name_text + '_gt_odom.csv'), 
                                        "world", "robot", CoordinateFrame.FLU, True, None, ts_in_ns=True)

        # Calculate RMS ATE, among other metrics
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True, axes_interval=500)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'], "\n")

        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])


if __name__ == "__main__":
    main()
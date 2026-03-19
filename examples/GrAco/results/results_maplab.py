from decimal import Decimal
import getpass
import numpy as np
import os
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, PathData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R
from pathlib import Path


# =============================================================================
# CONFIGURATION - Modify this section to add/remove robots
# =============================================================================
# Each robot entry: (robot_name, mission_id)
# The mission_id is used to filter the maplab CSV for this specific robot
ROBOTS = [
    ("ground1", " d7e9b9c4a3619e180b00000000000000")
    # ("Drone2", " 0acba9b3718491180b00000000000000"),
    # ("Drone1", " <mission-id-here>"),
]

DATASET_NAMES = ["V1.0"]

# Base path configuration
def get_base_path(user: str) -> str:
    return f'/media/{user}/T7/GT/SLAM/Data/GrAco_dataset'
# =============================================================================


def main():
    user = getpass.getuser()
    base_path = get_base_path(user)

    for dataset_name in DATASET_NAMES:
        robot_names = [robot[0] for robot in ROBOTS]

        # Load the estimated odometry data for each robot
        est_data_lst: list[OdometryData] = []
        for robot_name, mission_id in ROBOTS:
            est_path = f'{base_path}/{dataset_name}/results/maplab_results/merged_map/vertex_poses_velocities_biases.csv'
            est_data = OdometryData.from_csv(
                est_path, "odom", 'base_link', CoordinateFrame.NED, True,
                [0, 3, 4, 5, 6, 7, 8, 9],
                filter=(' mission-id', mission_id)
            )
            est_data.timestamps = est_data.timestamps / Decimal('1e9')  # Convert from ns to s
            est_data_lst.append(est_data)

        # Visualize initial estimated data
        est_data_lst[0].visualize([], [f"{robot_names[0]} Maplab Results"], 10, 40)


        # Load the ground truth data for each robot
        gt_data_lst: list[OdometryData] = []
        for robot_name, _ in ROBOTS:
            input_path = Path(base_path).absolute()
            gt_data = OdometryData.from_csv(input_path / dataset_name  / robot_name / 'odom' / 'odom.csv', robot_name + '/odom', robot_name + '/ground_truth/base_link', CoordinateFrame.NED, True)
            gt_data.to_FLU_frame()
            gt_data_lst.append(gt_data)

        # Make the timestamps match
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)

        # Since positions are in FLU but orientations are in NED rotated to FLU, lets fix that
        R_NED = np.array([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]])
        R_NED_Q = R.from_matrix(R_NED)

        # Apply coordinate transformations to individual estimated trajectories
        for est_data in est_data_lst:
            est_data._ori_apply_rotation(R_NED_Q.inv())
            est_data._ori_change_of_basis(R_NED_Q)
            est_data.frame = CoordinateFrame.FLU



        # 1. Compare individual robots with their ground truths
        for est_data, gt_data, name in zip(est_data_lst, gt_data_lst, robot_names):
            print(f"\n{'='*50}")
            print(f"Individual Results for {name}")
            print(f"{'='*50}")
            est_data.visualize([gt_data], [f"{name} Maplab Results", f"{name} Ground Truth"], [10, 10], [40, 1000])

            metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
            print(f"{name} RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
            print(f"{name} RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
            print(f"{name} RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
            print(f"{name} RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

        # 2. Compare combined results
        combined_name = "+".join(robot_names)
        print(f"\n{'='*50}")
        print(f"Combined Results ({combined_name})")
        print(f"{'='*50}")

        # Make the timestamps match and then concatenate
        if len(ROBOTS) > 1:
            est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
            est_data_combined: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
            gt_data_combined: PathData = PathData.concatenate_PathData(gt_data_lst)

            est_data_combined.visualize([gt_data_combined], [f"{combined_name} Maplab Results", "Ground Truth"], [10, 10], [40, 1000])

            metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data_combined, est_data_combined, max_diff=0.1, visualize=True)
            print("Combined RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
            print("Combined RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
            print("Combined RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
            print("Combined RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
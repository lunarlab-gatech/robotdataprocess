from decimal import Decimal
import numpy as np
import os
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, PathData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():  
    dataset_names = ["V2.1.0"]
    robot_names = ["Husky1", "Drone1"]

    # Do it for all files, robots, and datasets
    for dataset_name in dataset_names:
            
        # Load the odometry data
        est_data_robot1 = OdometryData.from_csv('/media/dbutterfield3/T73/Hercules_datasets/' + dataset_name + '/results/slideslam/' + robot_names[0] + '/trajectory.csv', "odom", 'base_link', CoordinateFrame.NED, True, [7, 0, 1, 2, 6, 3, 4, 5])
        est_data_robot2 = OdometryData.from_csv('/media/dbutterfield3/T73/Hercules_datasets/' + dataset_name + '/results/slideslam/' + robot_names[1] + '/trajectory.csv', "odom", 'base_link', CoordinateFrame.NED, True, [7, 0, 1, 2, 6, 3, 4, 5], reorder_data=True)
        est_data_lst: list[OdometryData] = [est_data_robot1, est_data_robot2]
        est_data_robot1.visualize([est_data_robot2], [robot_names[0] + " Results", robot_names[1] + " Results"], 10, 40)

        # Load the ground truth data
        gt_data_robot1 = OdometryData.from_csv('/media/dbutterfield3/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names[0] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_robot2 = OdometryData.from_csv('/media/dbutterfield3/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names[1] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_lst: list[OdometryData] = [gt_data_robot1, gt_data_robot2]

        # # Make the timestamps match and then concatenate
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
        est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # # Calculate RMS ATE, among other metrics
        metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
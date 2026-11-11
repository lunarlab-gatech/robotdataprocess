import getpass
import numpy as np
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R
from utils import LoadDataResult, print_errors, plot_GT_vs_est_on_image

def load_data_GAC_Mapping(dataset_name, experiment_name, robot0_name, robot1_name) -> LoadDataResult:

    # Instantiate a class to hold the results
    data = LoadDataResult()

    # Load the estimated data
    user = getpass.getuser()
    data.est_data_robot0 = OdometryData.from_tum('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/results/GAC-Mapping/' 
                                    + experiment_name + '/0.txt', "map", 'robot0', CoordinateFrame.ENU)
    data.est_data_robot1 = OdometryData.from_tum('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/results/GAC-Mapping/' 
                                    + experiment_name + '/1.txt', "map", 'robot1', CoordinateFrame.ENU)
    data.est_data_lst = [data.est_data_robot0, data.est_data_robot1]

    # For some reason, GAC-Mapping orientation is 180° rotated around the axes pointed equally in the YZ direction
    # Rotating to match the GT orientation. If we use rotation results, should found out why this is...
    H_rotation = R.from_rotvec(np.pi * np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)]))
    data.est_data_robot0._ori_apply_rotation_left_side(H_rotation)
    data.est_data_robot1._ori_apply_rotation_left_side(H_rotation)

    # Load the ground truth data
    data.gt_data_robot0 = OdometryData.from_tum('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                robot0_name + '/' + robot0_name + '.txt', "world", "robot0", CoordinateFrame.ENU)
    data.gt_data_robot1 = OdometryData.from_tum('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                robot1_name + '/' + robot1_name + '.txt', "world", "robot1", CoordinateFrame.ENU)
    data.gt_data_lst = [data.gt_data_robot0, data.gt_data_robot1]
    return data

def main():  
    dataset_name = "V1.0"
    experiment_name = "A07-A08"
    
    # Map experiment names to the robot names
    exp_to_robots_map = {
        "A07": "aerial-07",
        "A08": "aerial-08"
    }

    # Get robot0 name and robot1 name
    robot0_name = exp_to_robots_map[experiment_name[0:3]]
    robot1_name = exp_to_robots_map[experiment_name[-3:]]
        
    # Load the data
    data: LoadDataResult = load_data_GAC_Mapping(dataset_name, experiment_name, robot0_name, robot1_name)

    # Calculate individual RMS ATE, among other metrics
    print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(data.gt_data_robot0, data.est_data_robot0, max_diff=0.1, visualize=True)
    print_errors(metrics_dictionary)

    print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(data.gt_data_robot1, data.est_data_robot1, max_diff=0.1, visualize=True)
    print_errors(metrics_dictionary)

    # Make the timestamps match and then concatenate
    data.est_data_lst, data.gt_data_lst = PathData.make_start_and_end_times_match(data.est_data_lst, data.gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(data.est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(data.gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True)
    print_errors(metrics_dictionary)

    # Visualize the result on the dataset map
    plot_GT_vs_est_on_image(data, est_data_align, gt_data_align, dataset_name, robot0_name, robot1_name)

if __name__ == "__main__":
    main()
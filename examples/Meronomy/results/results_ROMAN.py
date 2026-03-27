from decimal import Decimal
import getpass
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from utils import LoadDataResult, print_errors, plot_GT_vs_est_on_image

MERONOMY_CROP_TIMES = {
    "V1.0": {
        "Husky1": (Decimal('1.0'), None),
        "Drone1": (Decimal('1.0'), None),
    },
    "V1.1": {
        "Husky1": (Decimal('1.0'), None),
        "Drone1": (Decimal('10.0'), None),
    },
    "V1.2": {
        "Husky1": (Decimal('1.0'), None),
        "Drone1": (Decimal('1.0'), None),
    },
}

def load_data_ROMAN(dataset_name, label_name: str, run_name, robot0_name, robot1_name) -> LoadDataResult:

    # Instantiate a class to hold the results
    data = LoadDataResult()

    # Load the estimated data
    user = getpass.getuser()
    run_path = '/home/' + user + '/Research/ROMAN_DEVEL/results/Meronomy_' + dataset_name + '_' + label_name + '/' + run_name + '/offline_rpgo/'
    data.est_data_robot0 = OdometryData.from_csv(run_path + robot0_name + '.csv', "map", 'robot0', CoordinateFrame.ENU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    data.est_data_robot1 = OdometryData.from_csv(run_path + robot1_name + '.csv', "map", 'robot0', CoordinateFrame.ENU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    data.est_data_lst = [data.est_data_robot0, data.est_data_robot1]

    # Load the ground truth data
    dataset_path = '/media/' + user + '/T73/Meronomy_datasets/' + dataset_name + '/data/'
    data.gt_data_robot0 = OdometryData.from_txt(dataset_path + robot0_name + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    data.gt_data_robot0.to_coordinate_frame(CoordinateFrame.FLU)
    data.gt_data_robot1 = OdometryData.from_txt(dataset_path + robot1_name + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    data.gt_data_robot1.to_coordinate_frame(CoordinateFrame.FLU)
    data.gt_data_lst = [data.gt_data_robot0, data.gt_data_robot1]
    return data

def main():  
    dataset_name = "V1.1"
    run_name = '17th'
    label_name = 'MeronomyGraph'
    robot_names = ["Husky1", "Drone1"]

    # Get robot0 name and robot1 name
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]
    
    # Load the data
    data: LoadDataResult = load_data_ROMAN(dataset_name, label_name, run_name, robot0_name, robot1_name)

    # Crop the data
    robot0_crop_times = MERONOMY_CROP_TIMES[dataset_name][robot0_name]
    data.est_data_robot0.crop_data(robot0_crop_times[0], robot0_crop_times[1])
    data.gt_data_robot0.crop_data(robot0_crop_times[0], robot0_crop_times[1])

    robot1_crop_times = MERONOMY_CROP_TIMES[dataset_name][robot1_name]
    data.est_data_robot1.crop_data(robot1_crop_times[0], robot1_crop_times[1])
    data.gt_data_robot1.crop_data(robot1_crop_times[0], robot1_crop_times[1])


    # Calculate individual RMS ATE, among other metrics
    print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(data.gt_data_robot0, data.est_data_robot0, max_diff=0.1, visualize=False)
    print_errors(metrics_dictionary)

    print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(data.gt_data_robot1, data.est_data_robot1, max_diff=0.1, visualize=False)
    print_errors(metrics_dictionary)

    # Make the timestamps match and then concatenate
    data.est_data_lst, data.gt_data_lst = PathData.make_start_and_end_times_match(data.est_data_lst, data.gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(data.est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(data.gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True)
    print_errors(metrics_dictionary)

    # Visualize the result on the dataset map
    plot_GT_vs_est_on_image(data, est_data_align, gt_data_align, dataset_name, robot0_name, robot1_name)

if __name__ == "__main__":
    main()
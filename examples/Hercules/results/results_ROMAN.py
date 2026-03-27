import getpass
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

def main():  
    dataset_name = "V2.3.AC"
    method = "MeronomyGraph"
    run_name = '27th'
    robot_names = ["Husky1", "Husky2"]

    # Get robot0 name and robot1 name
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]
    
    # Load the estimated data
    user = getpass.getuser()
    est_data_robot0 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method + '/' + run_name+ '/offline_rpgo/' + robot0_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_robot1 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method + '/' + run_name+ '/offline_rpgo/' + robot1_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_lst: list[OdometryData] = [est_data_robot0, est_data_robot1]

    # Load the ground truth data
    gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot0_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
    gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot1_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
    gt_data_lst: list[OdometryData] = [gt_data_robot0, gt_data_robot1]

    # Calculate individual RMS ATE, among other metrics
    print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    # Seperate the aligned trajectories into their single-robot forms
    gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
    gt_data_align_robot0 = gt_data_align_list[0]
    gt_data_align_robot1 = gt_data_align_list[1]

    est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
    est_data_align_robot0 = est_data_align_list[0]
    est_data_align_robot1 = est_data_align_list[1]

    # Get environment image path
    image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/environment.png'
    if dataset_name in "V2.3.AP":  x_edge = 350
    elif dataset_name in "V2.4.C": x_edge = 300
    elif dataset_name in "V2.3.AC": x_edge = 500
    elif dataset_name in "V2.4.F": x_edge = 150
    else:
        raise RuntimeError(f"x_edge not defined for {dataset_name}.")

    # Define the mapping from robot name to color and robot_name to new name
    name_map: dict = {
        "Husky1": "UGV1",
        "Husky2": "UGV2",
        "Drone1": "UAV1",
        "Drone2": "UAV2"
    }
    robot_name_to_color: dict = {
        "UGV1": "#1EE15F",
        "UGV2": "#E11E28",
        "UAV1": "#F0F02A",
        "UAV2": "#1B0ED5",
    }

    # Plot the results in 2D (Configuration for Figure 10)
    dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
    isGTList =  [                False,                 True,                  False,                  True]
    nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
    colorList = [robot_name_to_color[name] for name in nameList]
    PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True, 
                       background_image_path=image_path, background_image_x_edge=x_edge,
                       save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')
    
    # Configuration for Figure 2
    # PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=4.0, show_grid=False, 
    #                     background_image_path=image_path, background_image_x_edge=x_edge, legend=False, no_border=True,
    #                     save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

if __name__ == "__main__":
    main()
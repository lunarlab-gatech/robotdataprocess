import getpass
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

def main():  
    dataset_name = "V1.0"
    run_name = 'latest'
    robot_names = ["aerial-07", "aerial-08"]

    # Get robot0 name and robot1 name
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]
    
    # Load the estimated data
    user = getpass.getuser()
    est_data_robot0 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/GrAco_' + dataset_name + '_aerial/' + run_name+ '/offline_rpgo/' + robot0_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_robot1 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/GrAco_' + dataset_name + '_aerial/' + run_name+ '/offline_rpgo/' + robot1_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_lst: list[OdometryData] = [est_data_robot0, est_data_robot1]

    # Load the ground truth data
    gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                           robot0_name + '/' + robot0_name + '.txt', "world", "robot", CoordinateFrame.ENU, 
                                             False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                           robot1_name + '/' + robot1_name + '.txt', "world", "robot", CoordinateFrame.ENU,  
                                           False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_lst: list[OdometryData] = [gt_data_robot0, gt_data_robot1]

    # Calculate individual RMS ATE, among other metrics
    print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Standard Deviation RTE: ", metrics_dictionary['RPE']['translation_part']['std'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])
    print("Standard Deviation RTE Rotation Angle (Rad): ", metrics_dictionary['RPE']['rotation_angle_rad']['std'])

    print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, run_name, "============")
    metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Standard Deviation RTE: ", metrics_dictionary['RPE']['translation_part']['std'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])
    print("Standard Deviation RTE Rotation Angle (Rad): ", metrics_dictionary['RPE']['rotation_angle_rad']['std'])

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Standard Deviation RTE: ", metrics_dictionary['RPE']['translation_part']['std'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])
    print("Standard Deviation RTE Rotation Angle (Rad): ", metrics_dictionary['RPE']['rotation_angle_rad']['std'])

    # Seperate the aligned trajectories into their single-robot forms
    gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
    gt_data_align_robot0 = gt_data_align_list[0]
    gt_data_align_robot1 = gt_data_align_list[1]

    est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
    est_data_align_robot0 = est_data_align_list[0]
    est_data_align_robot1 = est_data_align_list[1]

    # Get environment image path
    image_path = '/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/environment.png'
    x_edge = 691.216296

    # Define the mapping from robot name to color and robot_name to new name
    robot_name_to_color: dict = {
        "ground-01": "#D61AD0",
        "ground-06": "#12EF49",
        "aerial-07": "#1A46D6",
        "aerial-08": "#E8EF12",
    }

    # Plot the results in 2D
    dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
    isGTList =  [                False,                 True,                  False,                  True]
    nameList =  [robot0_name, robot0_name, robot1_name, robot1_name]
    colorList = [robot_name_to_color[name] for name in nameList]
    PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=3.0, show_grid=False, 
                       background_image_path=image_path, background_image_x_edge=x_edge, gt_color_lightness_range_val=12,
                       background_image_extent_offsets=(55, 80), no_border=True,
                       save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

if __name__ == "__main__":
    main()
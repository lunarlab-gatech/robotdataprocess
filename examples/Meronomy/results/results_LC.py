import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

def calculate_LC_errors_ROMAN(dataset_name: str, label_name: str, run_name: str, robot_names: List, all_error_dicts: list, all_inlier_masks: list):

    # Get robot name pair for this configuration
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/Meronomy_' + dataset_name + '_' + label_name + '/' + run_name)

    # Load Loop Closure data
    lc_data = LoopClosureData.from_json(run_folder / 'align' / (robot_names[0] + '_' + robot_names[1]) / 'align.json')
    lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' /'inlier_lc_inter_robot.g2o', run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt', 
                                              names_override={"a": robot_names[0], "b": robot_names[1]})

    # Round timestamps to allow proper matching
    lc_data.round_timestamps(4)
    lc_data_inlier.round_timestamps(4)

    # Merge the datas so we know which are inliers vs. outliers
    lc_data.label_inliers_via_other_LoopClosureData(lc_data_inlier)

    # Load the GT data for both robots
    dataset_path = '/media/' + user + '/T73/Meronomy_datasets/' + dataset_name + '/data/'
    gt_data_robot0 = OdometryData.from_txt(dataset_path + robot_names[0] + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    gt_data_robot0.to_coordinate_frame(CoordinateFrame.FLU)
    gt_data_robot1 = OdometryData.from_txt(dataset_path + robot_names[1] + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    gt_data_robot1.to_coordinate_frame(CoordinateFrame.FLU)
    gt_data_dict: dict[str, OdometryData] = {robot_names[0]: gt_data_robot0, robot_names[1]: gt_data_robot1}

    # Calculate the errors for the loop closures
    all_error_dicts.append(lc_data.calculate_errors(gt_data_dict))
    all_inlier_masks.append(lc_data.detected_inliers)

def main():
    # ====================== ROMAN ===========================
    # Set dataset configuration
    run_names = ["17th", "17th", "25th", "25th", "30th", "1st"]
    label_names = ["MeronomyGraph", "ROMAN", "MeronomyGraph", "ROMAN", "MeronomyGraph", "MeronomyGraph"]
    dataset_names = ["V1.1"] * len(run_names)
    robot_names = ["Husky1", "Drone1"]
    run_display_names = ["ROMAN Mapper" if r == "17th" else "New Mapper" for r in run_names]
    title_names = [label + " - " + run for label, run in zip(label_names, run_display_names)]

    # Calculate lc errors for each run and robot pair for ROMAN
    errors_list = []
    inliers_list = []
    for dataset_name, label_name, run_name in zip(dataset_names, label_names, run_names):
        calculate_LC_errors_ROMAN(dataset_name, label_name, run_name, robot_names, errors_list, inliers_list)

    # Visualize the results
    LoopClosureData.visualize_error_scatter(errors_list, title_names, inliers_list, max_rotation_frac=1.0, max_translation_frac=1.0,
                                            trans_err_in_target=1.0, show_plots=False, rot_err_in_target=5.0, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')

if __name__ == "__main__":
    main()
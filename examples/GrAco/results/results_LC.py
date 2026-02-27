import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

def calculate_LC_errors_ROMAN(dataset_name: str, run_name: str, robot_names: List, all_error_dicts: dict, all_inlier_masks: dict):

    # Get robot name pair for this configuration
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/GrAco_' + dataset_name + '/' + run_name)

    # Load Loop Closure data
    lc_data = LoopClosureData.from_json(run_folder / 'align' / (robot_names[0] + '_' + robot_names[1]) / 'align.json')
    lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' /'inlier_lc_inter_robot.g2o', run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt', names_override={"a": robot_names[0], "b": robot_names[1]})

    # Round timestamps to allow proper matching
    lc_data.round_timestamps(4)
    lc_data_inlier.round_timestamps(4)

    # Merge the datas so we know which are inliers vs. outliers
    lc_data.label_inliers_via_other_LoopClosureData(lc_data_inlier)

    # Load the GT data for both robots
    gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                            robot_names[0] + '/' + robot_names[0] + '.txt', "world", "robot", CoordinateFrame.ENU, 
                            False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                            robot_names[1] + '/' + robot_names[1] + '.txt', "world", "robot", CoordinateFrame.ENU,  
                            False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_dict: dict[str, OdometryData] = {robot_names[0]: gt_data_robot0, robot_names[1]: gt_data_robot1}

    gt_data_robot0.visualize_3D([gt_data_robot1], ["ground-06", "aerial-08"])
    
    # Calculate the errors for the loop closures
    all_error_dicts[run_name] = lc_data.calculate_errors(gt_data_dict)
    all_inlier_masks[run_name] = lc_data.detected_inliers

def calculate_LC_errors_GACMapping(dataset_name: str, run_name: str, robot_names: str, all_error_dicts: dict, all_inlier_masks: dict):

    # Get robot name pair for this configuration
    user = getpass.getuser()
    run_folder = Path('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/results/GAC-Mapping/' + run_name)

    # Load Loop Closure data
    lc_data = LoopClosureData.from_json(run_folder / 'robot0_robot1_inter_loop_closures.json', names_override={"0": robot_names[0], "1": robot_names[1]})

    # Round timestamps to allow proper matching
    lc_data.round_timestamps(4)

    # Load the GT data for both robots
    gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                            robot_names[0] + '/' + robot_names[0] + '.txt', "world", "robot", CoordinateFrame.ENU, 
                            False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                            robot_names[1] + '/' + robot_names[1] + '.txt', "world", "robot", CoordinateFrame.ENU,  
                            False, [0, 1, 2, 3, 7, 4, 5, 6])
    gt_data_dict: dict[str, OdometryData] = {robot_names[0]: gt_data_robot0, robot_names[1]: gt_data_robot1}
    
    # Calculate the errors for the loop closures
    all_error_dicts[run_name] = lc_data.calculate_errors(gt_data_dict)
    all_inlier_masks[run_name] = None

def main():
    # ====================== ROMAN ===========================
    # Set dataset configuration
    run_names = ["roman-run-fb01c1ca", "meronomy-run-fb01c1ca"]
    dataset_names = ["V1.0"] * len(run_names)
    robot_names = ["ground-06", "aerial-08"]

    # Calculate lc errors for each run and robot pair for ROMAN
    all_error_dicts = {}
    all_inlier_masks = {}
    for dataset_name, run_name in zip(dataset_names, run_names):
        calculate_LC_errors_ROMAN(dataset_name, run_name, robot_names, all_error_dicts, all_inlier_masks)

    # ====================== GAC-Mapping  ===========================
    # Set dataset configuration
    # dataset_name = "V1.0"
    # run_name = "A07-A08"

    # # Calculate lc errors for each run and robot pair for ROMAN
    # calculate_LC_errors_GACMapping(dataset_name, run_name, robot_names, all_error_dicts, all_inlier_masks)

    # Visualize the results
    errors_list = list(all_error_dicts.values())
    inliers_list = list(all_inlier_masks.values())
    label_names = ["ROMAN", "MeronomyGraph"]
    LoopClosureData.visualize_error_scatter(errors_list, label_names, inliers_list, max_rotation_frac=1.0, max_translation_frac=1.0, trans_err_in_target=3.0, show_plots=False, rot_err_in_target=6.0, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')

if __name__ == "__main__":
    main()
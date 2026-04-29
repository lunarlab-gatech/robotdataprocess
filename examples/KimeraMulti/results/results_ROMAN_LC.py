import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

def calculate_LC_errors_ROMAN(run_name: str, robot_names: List, all_error_dicts: list):

    # Get robot name pair for this configuration
    user = getpass.getuser()
    run_folder = Path('/home/') / user / 'Research/roman/kimera_multi_output' / run_name

    # Load Loop Closure data
    lc_data = LoopClosureData.from_json(run_folder / 'align' / (robot_names[0] + '_' + robot_names[1]) / 'align.json')

    # Round timestamps to allow proper matching
    lc_data.round_timestamps(4)

    # Load the GT data for both robots
    dataset_path = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data' / 'ground_truth' / '1207'
    gt_data_robot0 = OdometryData.from_csv(dataset_path / (robot_names[0] + '_gt_odom.csv'), 'world', 'robot', 
                                           CoordinateFrame.FLU, True, None, ts_in_ns=True)
    gt_data_dict: dict[str, OdometryData] = {robot_names[0]: gt_data_robot0}

    # Calculate the errors for the loop closures
    all_error_dicts.append(lc_data.calculate_errors(gt_data_dict))

def main():
    # ====================== ROMAN ===========================
    # Set dataset configuration
    run_names =  ["Tunnel1"]
    robot_names = ["acl_jackal2", "acl_jackal2"]

    # Calculate lc errors for each run and robot pair for ROMAN
    errors_list = []
    for run_name in run_names:
        calculate_LC_errors_ROMAN(run_name, robot_names, errors_list)

    # Visualize the results
    LoopClosureData.visualize_error_scatter(errors_list, run_names, None, max_rotation_frac=1.0, max_translation_frac=1.0, trans_err_in_target=1.0, show_plots=False, rot_err_in_target=5.0, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')

if __name__ == "__main__":
    main()
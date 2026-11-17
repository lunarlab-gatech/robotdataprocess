import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List, Union

def calculate_LC_errors_maplab(dataset_name: str, file_path: Union[Path, str], robot_names: List, lc_data_list: list):

    # Load Loop Closure data (NOTE: The names override is hardcoded)
    lc_data = LoopClosureData.from_maplab_json(file_path, names_override={"3f67cb5349fea5180b00000000000000": "Husky1", "99a8765349fea5180b00000000000000": "Husky2"})

    # Round timestamps to allow proper matching
    lc_data.round_timestamps(4)
    lc_data.prune_intra_robot_loop_closures() # NOTE: This file only evaluates inter-robot loop closures.

    # Load the GT data for both robots
    user = getpass.getuser()
    dataset_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/'
    gt_data_robot0 = OdometryData.from_txt(dataset_path + robot_names[0] + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    gt_data_robot0.to_coordinate_frame(CoordinateFrame.FLU)
    gt_data_robot1 = OdometryData.from_txt(dataset_path + robot_names[1] + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
    gt_data_robot1.to_coordinate_frame(CoordinateFrame.FLU)
    gt_data_dict: dict[str, OdometryData] = {robot_names[0]: gt_data_robot0, robot_names[1]: gt_data_robot1}

    # Calculate the errors for the loop closures
    lc_data.calculate_errors(gt_data_dict)
    lc_data.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)
    lc_data_list.append(lc_data)

def main():
    # ====================== ROMAN ===========================
    # Set dataset configuration
    dataset_name = "V2.4.C"
    robot_names = ["Husky1", "Husky2"]
    title_names = ["Maplab"]
    file_path = "/home/dbutterfield3/Downloads/lc.json"

    # Calculate lc errors for each run and robot pair for ROMAN
    lc_data_list = []
    calculate_LC_errors_maplab(dataset_name, file_path, robot_names, lc_data_list)

    # Visualize the results
    LoopClosureData.visualize_error_scatter(lc_data_list, title_names, None, max_rotation_frac=1.0, max_translation_frac=1.0, show_plots=False, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')

if __name__ == "__main__":
    main()
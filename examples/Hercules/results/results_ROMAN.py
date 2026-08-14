from decimal import Decimal
import getpass
import sys
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from utils.ROMAN import run_ROMAN_evaluation

def load_gt_data_ROMAN(dataset_name: str, robot_names: List) -> List[OdometryData]:
    """
    Load ground truth trajectories for a set of robots from poseGT.csv.

    Returns:
        List of OdometryData in the same order as robot_names, in FLU frame.
    """
    user = getpass.getuser()
    gt_data_list: List[OdometryData] = []
    for rn in robot_names:
        pose_data = OdometryData.from_txt('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/' + rn + '/pose_world_frame.txt',
                     rn + '/odom', rn + '/ground_truth/base_link', CoordinateFrame.NED, False)
        pose_data.to_coordinate_frame(CoordinateFrame.FLU)

        if dataset_name == "V2.4.C":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('382.85'), Decimal('390.90'), Decimal('1100.00'), Decimal('1190.35')]
        elif dataset_name == "V2.3.AP":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('772.15'), Decimal('741.45'), Decimal('1121.80'), Decimal('1193.80')]
        elif dataset_name == "V2.3.AC":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('1125.00'), Decimal('1118.80'), Decimal('1025.50'), Decimal('892.60')]
        elif dataset_name == "V2.4.F":
            robot_crop_start_times = [Decimal('35.05'), Decimal('34.60'), Decimal('27.45'), Decimal('31.50')]
            robot_crop_end_times = [Decimal('575.55'), Decimal('762.35'), Decimal('898.10'), Decimal('906.85')]
        elif dataset_name == "SmallTownSequence":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [None, None, None, None]
        else:
            raise ValueError("Crop times not specified for this dataset number.")

        robot_name_to_index: dict = {"Husky1": 0, "Husky2": 1, "Drone1": 2, "Drone2": 3}
        pose_data.crop_data(robot_crop_start_times[robot_name_to_index[rn]],
                            robot_crop_end_times[robot_name_to_index[rn]])
        gt_data_list.append(pose_data)

    return gt_data_list

def main():
    """
    Generate all evaluation figures and tables for the HERCULES dataset.

    See :func:`utils.results_ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    run_names = ["ROMAN_O", "MG_TS"] # "ROMAN_O"
    dataset_name = "V2.4.C"

    # Environment image / robot display config
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/environment.png'
    if dataset_name in "V2.3.AP":  x_edge = 350
    elif dataset_name in "V2.4.C": x_edge = 300
    elif dataset_name in "V2.3.AC": x_edge = 500
    elif dataset_name in "V2.4.F": x_edge = 150
    elif dataset_name in "SmallTownSequence": x_edge = 150
    else:
        raise RuntimeError(f"x_edge not defined for {dataset_name}.")

    name_map: Dict = {
        "Husky1": "UGV1",
        "Husky2": "UGV2",
        "Drone1": "UAV1",
        "Drone2": "UAV2"
    }
    robot_name_to_color: Dict = {
        "UGV1": "#1EE15F",
        "UGV2": "#E11E28",
        "UAV1": "#F0F02A",
        "UAV2": "#1B0ED5",
    }
    viz_config = {
        "image_path": image_path,
        "x_edge": x_edge,
        "name_map": name_map,
        "robot_name_to_color": robot_name_to_color,
    }

    figures_base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures')
    roman_root = Path('/home/dbutterfield3/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    run_ROMAN_evaluation(roman_root, "hercules", dataset_name, run_names, all_robots, critical_invocation_params,
                         figures_base_dir, load_gt_data_ROMAN, viz_config, ate_threshold_m=20.0)

if __name__ == "__main__":
    main()

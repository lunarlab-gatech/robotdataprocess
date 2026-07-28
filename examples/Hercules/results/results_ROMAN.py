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
    return [
        OdometryData.from_csv(
            '/media/' + user + '/T73/Hercules_datasets/' + dataset_name +
            '/extract/files_for_roman_baseline/' + rn + '/poseGT.csv',
            'world', 'robot', CoordinateFrame.FLU, True, None)
        for rn in robot_names
    ]

def main():
    """
    Generate all evaluation figures and tables for the HERCULES dataset.

    See :func:`utils.results_ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    run_names = ["ROMAN", "ROMAN_NM", "MG_NONM", "MG"] # "ROMAN_O"
    dataset_name = "V2.4.C"

    # Environment image / robot display config
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/environment.png'
    if dataset_name in "V2.3.AP":  x_edge = 350
    elif dataset_name in "V2.4.C": x_edge = 300
    elif dataset_name in "V2.3.AC": x_edge = 500
    elif dataset_name in "V2.4.F": x_edge = 150
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

    run_ROMAN_evaluation("hercules", dataset_name, run_names, all_robots, figures_base_dir,
                         load_gt_data_ROMAN, viz_config)

if __name__ == "__main__":
    main()

import getpass
import itertools
import re
import sys
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.ROMAN import run_ROMAN_evaluation

def load_gt_data_ROMAN(dataset_name: str, robot_names: List) -> List[OdometryData]:
    """
    Load ground truth trajectories for a set of robots from <robot_name>_gt_odom.csv.

    Returns:
        List of OdometryData in the same order as robot_names, in FLU frame.
    """

    user = getpass.getuser()
    dataset_number = re.search(r'\d{4}', dataset_name).group()
    gt_data: List[OdometryData] = []
    for rn in robot_names:
        data = OdometryData.from_csv('/media/' + user + '/T73/Kimera-Multi_Dataset/data/ground_truth/'
                              + dataset_number + '/' + rn + '_gt_odom.csv', 'world', 'robot',
                              CoordinateFrame.FLU, True, None, ts_in_ns=True)
        gt_data.append(data)
    return gt_data

def main():
    """
    Generate all evaluation figures and tables for the Kimera-Multi dataset.

    See :func:`utils.results_ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """

    all_robots = ["acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek"]
    robot_groups = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN_O", "MG_TS", "MG"]
    dataset_name = "campus_tunnels_1207_compressed"

    # Environment image / robot display config
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/Kimera-Multi_Dataset/data/' + dataset_name + '/environment.jpeg'
    x_edge: float = 0.0

    robot_name_to_color: Dict = {
        "acl_jackal": "#FFA501",
        "acl_jackal2": "#FF0101",
        "sparkal1": "#008000",
        "sparkal2": "#0014FF",
        "hathor": "#00FFFF",
        "thoth": "#FF00FF",
        "apis": "#808080",
        "sobek": "#000000",
    }
    viz_config = {
        "image_path": image_path,
        "x_edge": x_edge,
        "robot_name_to_color": robot_name_to_color,
    }

    figures_base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures')
    roman_root = Path('/home/dbutterfield3/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    run_ROMAN_evaluation(roman_root, "kimera_multi", dataset_name, run_names, robot_groups, critical_invocation_params,
                         figures_base_dir, load_gt_data_ROMAN, viz_config, ate_threshold_m=10.0)

if __name__ == "__main__":
    main()

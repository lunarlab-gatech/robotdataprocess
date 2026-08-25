import getpass
import re
import sys
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.ROMAN import run_ROMAN_evaluation

DATASET_ROBOT_GROUPS: Dict[str, List[Tuple[str, ...]]] = {
    "campus_tunnels_1207_compressed": [
        ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek"),
        ("acl_jackal",),
        ("acl_jackal2",),
        ("sparkal1",),
        ("sparkal2",),
    ],
    # "campus_hybrid_1208_compressed": [
    #     ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek"),
    #     ("acl_jackal", "acl_jackal2", "sparkal1"),
    #     ("sparkal2", "hathor"),
    # ],
    # "campus_outdoor_1014_compressed": [
    #     ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth"),
    #     ("acl_jackal", "acl_jackal2"),
    # ],
}

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

    Evaluates the robot groups in DATASET_ROBOT_GROUPS, once per dataset sequence.

    See :func:`robotdataprocess.eval.ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """

    run_names = ["ROMAN_O", "MG_TS", "MG"]

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
        "image_path": None,
        "x_edge": None,
        "robot_name_to_color": robot_name_to_color,
    }

    figures_base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures')
    roman_root = Path('/home/dbutterfield3/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    for dataset_name, robot_groups in DATASET_ROBOT_GROUPS.items():
        run_ROMAN_evaluation(roman_root, "kimera_multi", dataset_name, run_names, robot_groups, critical_invocation_params,
                             figures_base_dir, load_gt_data_ROMAN, viz_config, ate_threshold_m=10.0)

if __name__ == "__main__":
    main()

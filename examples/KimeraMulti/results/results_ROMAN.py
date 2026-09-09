import getpass
import re
import sys
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.RobotGroup import RobotGroup
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator

DATASET_NAME = "kimera_multi"
TUNNELS_SEQ = "campus_tunnels_1207_compressed"
HYBRID_SEQ = "campus_hybrid_1208_compressed"
OUTDOOR_SEQ = "campus_outdoor_1014_compressed"

TUNNELS_ROBOTS = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek")
HYBRID_ROBOTS = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek")
OUTDOOR_ROBOTS = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth")

def load_gt_data_ROMAN(dataset_seq: str, robot_names: List) -> List[OdometryData]:
    """
    Load ground truth trajectories for a set of robots from <robot_name>_gt_odom.csv.

    Returns:
        List of OdometryData in the same order as robot_names, in FLU frame.
    """

    user = getpass.getuser()
    dataset_number = re.search(r'\d{4}', dataset_seq).group()
    gt_data: List[OdometryData] = []
    for rn in robot_names:
        data = OdometryData.from_csv('/media/' + user + '/T73/Kimera-Multi_Dataset/data/ground_truth/'
                              + dataset_number + '/' + rn + '_gt_odom.csv', 'world', 'robot',
                              CoordinateFrame.FLU, True, None, ts_in_ns=True)
        gt_data.append(data)
    return gt_data

def main():
    """
    Generate all evaluation figures and tables for the Kimera-Multi dataset, grouped the way the
    Kimera-Multi paper does -- by robot count, not by dataset sequence -- so each grouping call
    below spans whichever sequences it needs.

    See :meth:`SLAMEvaluator.run_evaluation` for the outputs produced.
    """

    run_names = ["ROMAN_O", "MG_TS_SM", "MG_SM"]

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

    # The four single-robot tunnel groups -- all from the same sequence.
    easy = SLAMEvaluator.make_robot_groups(DATASET_NAME, TUNNELS_SEQ,
        [("acl_jackal",), ("acl_jackal2",), ("sparkal1",), ("sparkal2",)])

    # The full-robot-set group from each sequence. Labeled explicitly: tunnels and hybrid share
    # the same 8 robots, so the default group_label would collide between them.
    medium = [
        RobotGroup(robots=TUNNELS_ROBOTS, dataset_name=DATASET_NAME, dataset_seq=TUNNELS_SEQ, label="tunnels"),
        RobotGroup(robots=HYBRID_ROBOTS, dataset_name=DATASET_NAME, dataset_seq=HYBRID_SEQ, label="hybrid"),
        RobotGroup(robots=OUTDOOR_ROBOTS, dataset_name=DATASET_NAME, dataset_seq=OUTDOOR_SEQ, label="outdoor"),
    ]

    # The three remaining 2-3 robot groups, spanning the hybrid and outdoor sequences.
    difficult = [
        RobotGroup(robots=("acl_jackal", "acl_jackal2", "sparkal1"), dataset_name=DATASET_NAME, dataset_seq=HYBRID_SEQ,
                   label=SLAMEvaluator.group_label(("acl_jackal", "acl_jackal2", "sparkal1"))),
        RobotGroup(robots=("sparkal2", "hathor"), dataset_name=DATASET_NAME, dataset_seq=HYBRID_SEQ,
                   label=SLAMEvaluator.group_label(("sparkal2", "hathor"))),
        RobotGroup(robots=("acl_jackal", "acl_jackal2"), dataset_name=DATASET_NAME, dataset_seq=OUTDOOR_SEQ,
                   label=SLAMEvaluator.group_label(("acl_jackal", "acl_jackal2"))),
    ]

    for output_name, robot_groups in [("Easy", easy), ("Medium", medium), ("Difficult", difficult)]:
        SLAMEvaluator.run_evaluation(roman_root, Path(DATASET_NAME) / output_name, run_names, robot_groups,
                                 critical_invocation_params, figures_base_dir, load_gt_data_ROMAN, viz_config, ate_threshold_m=10.0)

if __name__ == "__main__":
    main()

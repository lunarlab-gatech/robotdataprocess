import getpass
import itertools
import sys
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.RobotGroup import RobotGroupViz
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator

NAME_TO_FRAME_MAP: dict = {
    "drone": CoordinateFrame.FLU,
    "robotA": CoordinateFrame.UFL,
    "robotB": CoordinateFrame.UFL,
    "robotC": CoordinateFrame.FUR
}

def load_gt_data_ROMAN(dataset_seq: str, robot_names: List) -> List[OdometryData]:
    """
    Load ground truth trajectories for a set of robots from <robot_name>.txt.

    Returns:
        List of OdometryData in the same order as robot_names, in ENU frame.
    """

    user = getpass.getuser()
    gt_data: List[OdometryData] = []
    for rn in robot_names:
        data = OdometryData.from_txt('/media/' + user + '/T73/AirMuseum_dataset/' + dataset_seq + '/data/'
                              + rn + '/body_stamped_groundtruth.txt', 'world', 'robot',
                              CoordinateFrame.NONE, True, [0, 1, 2, 3, 7, 4, 5, 6])
        data.redefine_local_axes(NAME_TO_FRAME_MAP[rn], CoordinateFrame.FLU)
        gt_data.append(data)
    return gt_data

def make_viz_config() -> RobotGroupViz:
    """Builds the AirMuseum environment image / robot display config, shared across all its scenarios."""
    user = getpass.getuser()
    robot_name_to_color: Dict = {
        "drone": "#FFA501",
        "robotA": "#FF0101",
        "robotB": "#008000",
        "robotC": "#0014FF",
    }
    return RobotGroupViz( # TODO: This is off
        name_map=None,
        robot_name_to_color=robot_name_to_color,
        image_path='/media/' + user + '/T73/AirMuseum_dataset/environment.png',
        image_x_edge=39,
        image_extent_offsets=(-12.5, 3),
        yaw_rotation_deg=280.0,
    )

def main():
    """
    Generate all evaluation figures and tables for the AirMuseum dataset.

    See :meth:`SLAMEvaluator.run_evaluation` for the outputs produced.
    """

    all_robots = ["drone", "robotA", "robotB", "robotC"]
    robot_groups = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN_O_SM", "MG_TS_SM", "MG_SM"] # "MG_TS_SM", "MG_SM"
    dataset_seq = "Scenario3"
    viz_config = make_viz_config()

    user = getpass.getuser()
    figures_base_dir = Path('/home/' + user + '/Research/robotdataprocess/figures')
    roman_root = Path('/home/' + user + '/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    robot_groups = SLAMEvaluator.make_robot_groups("airmuseum", dataset_seq, robot_groups, viz_config)
    SLAMEvaluator.run_evaluation(roman_root, Path("airmuseum") / dataset_seq, run_names, robot_groups,
                             critical_invocation_params, figures_base_dir, load_gt_data_ROMAN, ate_threshold_m=10.0)

if __name__ == "__main__":
    main()

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
    Load ground truth trajectories for a set of robots from <robot_name>.txt.

    Returns:
        List of OdometryData in the same order as robot_names, in ENU frame.
    """

    name_to_frame_map: dict = {
        "drone": CoordinateFrame.FLU,
        "robotA": CoordinateFrame.UFL,
        "robotB": CoordinateFrame.UFL,
        "robotC": CoordinateFrame.FUR
    }
    
    user = getpass.getuser()
    gt_data: List[OdometryData] = []
    for rn in robot_names:
        data = OdometryData.from_txt('/media/' + user + '/T73/AirMuseum_dataset/' + dataset_name + '/data/'
                              + rn + '/body_stamped_groundtruth.txt', 'world', 'robot',
                              CoordinateFrame.FLU, True, [0, 1, 2, 3, 7, 4, 5, 6])
        data.redefine_local_axes(name_to_frame_map[rn], CoordinateFrame.FLU)
        gt_data.append(data)
    return gt_data

def main():
    """
    Generate all evaluation figures and tables for the AirMuseum dataset.

    See :func:`utils.results_ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """

    all_robots = ["drone", "robotA"]
    run_names = ["ROMAN_O"]
    dataset_name = "Scenario5"

    # Environment image / robot display config
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/AirMuseum_dataset/environment.png'
    x_edge: float = 39 # TODO: This is off

    robot_name_to_color: Dict = {
        "drone": "#FFA501",
        "robotA": "#FF0101",
        "robotB": "#008000",
        "robotC": "#0014FF",
    }
    viz_config = { # TODO: This is off
        "image_path": image_path,
        "x_edge": x_edge,
        "robot_name_to_color": robot_name_to_color,
        "background_image_extent_offsets": (-12.5, 3),
        "yaw_rotation_deg": 280.0,
    }

    figures_base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures')

    run_ROMAN_evaluation("airmuseum", dataset_name, run_names, all_robots, figures_base_dir, load_gt_data_ROMAN, viz_config)

if __name__ == "__main__":
    main()

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
    user = getpass.getuser()
    return [
        OdometryData.from_csv(
            '/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + rn + '/' + rn + '.txt',
            'world', 'robot', CoordinateFrame.ENU, False, [0, 1, 2, 3, 7, 4, 5, 6])
        for rn in robot_names
    ]

def main():
    """
    Generate all evaluation figures and tables for the GrAco dataset.

    See :func:`utils.results_ROMAN.run_ROMAN_evaluation` for the outputs produced.
    """

    all_robots = ["ground-06", "aerial-08"]
    run_names = ["ROMAN_O"]
    dataset_name = "V1.0"

    # Environment image / robot display config
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/environment.png'
    x_edge = 691.216296

    robot_name_to_color: Dict = {
        "ground-01": "#D61AD0",
        "ground-06": "#12EF49",
        "aerial-07": "#1A46D6",
        "aerial-08": "#E8EF12",
    }
    viz_config = {
        "image_path": image_path,
        "x_edge": x_edge,
        "robot_name_to_color": robot_name_to_color,
        "background_image_extent_offsets": (55, 80),
    }

    figures_base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures')

    run_ROMAN_evaluation("graco", dataset_name, run_names, all_robots, figures_base_dir, load_gt_data_ROMAN, viz_config)

if __name__ == "__main__":
    main()

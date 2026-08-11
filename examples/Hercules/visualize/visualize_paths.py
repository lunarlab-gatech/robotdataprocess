import getpass
import sys
from pathlib import Path
from robotdataprocess import OdometryData, PathData
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'results'))
from results_ROMAN import load_gt_data_ROMAN

def main():
    """
    Generate an interactive 2D video of the ground-truth trajectories for a
    robot pair on the HERCULES dataset.

    See :func:`PathData.visualize_2D_video` for the animation itself.
    """
    dataset_seq = "V2.3.AP"
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_seq + '/data/environment.png'
    if dataset_seq == "V2.3.AP": x_edge = 350
    elif dataset_seq == "V2.4.C": x_edge = 300
    elif dataset_seq == "V2.3.AC": x_edge = 500
    elif dataset_seq == "V2.4.F": x_edge = 150
    elif dataset_seq == "SmallTownSequence": x_edge = 150
    else:
        raise RuntimeError(f"x_edge not defined for {dataset_seq}.")

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

    gt_data_list: List[OdometryData] = load_gt_data_ROMAN(dataset_seq, robot_names)

    dataList = gt_data_list
    nameList = [name_map[rn] + " (GT)" for rn in robot_names]
    colorList = [robot_name_to_color[name_map[rn]] for rn in robot_names]

    output_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures/hercules') / dataset_seq / 'videos'
    PathData.visualize_2D_video(dataList, colorList, nameList, video_duration_sec=20.0, fps=30,
                                background_image_path=image_path, background_image_x_edge=x_edge,
                                show_grid=True, legend=True, glow_radius_px=10, fade_trail_decay=0.9, 
                                save_path=str(output_dir / 'gt_paths.mp4'))

if __name__ == "__main__":
    main()

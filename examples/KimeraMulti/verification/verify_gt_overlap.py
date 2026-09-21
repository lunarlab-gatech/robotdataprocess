import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "results"))
import results_ROMAN

from robotdataprocess import PathData

plt.rcParams['figure.figsize'] = (28, 20)  # visualize_2D creates its figure at this default size

def main():
    """
    Plots every robot pair's GT-only trajectories (no estimated data, no background image) for
    each of the three Kimera-Multi sequences, one pair at a time, shown interactively instead of
    saved -- to visually check whether each pair's ground truth actually overlaps in space before
    evaluating them together. Close a figure to advance to the next pair.
    """
    seqs_and_robots = [
        (results_ROMAN.TUNNELS_SEQ, results_ROMAN.TUNNELS_ROBOTS),
        (results_ROMAN.HYBRID_SEQ, results_ROMAN.HYBRID_ROBOTS),
        (results_ROMAN.OUTDOOR_SEQ, results_ROMAN.OUTDOOR_ROBOTS),
    ]
    viz_config = results_ROMAN.make_viz_config()

    for dataset_seq, robots in seqs_and_robots:
        gt_by_robot = dict(zip(robots, results_ROMAN.load_gt_data_ROMAN(dataset_seq, robots)))

        for robot_a, robot_b in itertools.combinations(robots, 2):
            print(f"Showing {dataset_seq}: {robot_a} vs {robot_b}...")
            PathData.visualize_2D(
                [gt_by_robot[robot_a], gt_by_robot[robot_b]], [True, True],
                [viz_config.robot_name_to_color[robot_a], viz_config.robot_name_to_color[robot_b]],
                [robot_a, robot_b],
                no_background=True, line_width=5.0, show_grid=True, gt_color_lightness_range_val=8,
                title=f"{dataset_seq}: {robot_a} vs {robot_b}",
            )


if __name__ == "__main__":
    main()

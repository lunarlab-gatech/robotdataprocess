import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "results"))
import results_MG

def max_pairwise_distance(points: np.ndarray) -> float:
    """
    Returns the largest 3D distance between any two of ``points`` (N, 3), comparing only convex
    hull vertices since the farthest pair always lies on the hull. ``QJ`` joggles the input so
    near-planar point sets (e.g. ground robots) still yield a hull.
    """
    hull = ConvexHull(points, qhull_options="QJ")
    return float(pdist(points[hull.vertices]).max())

def main():
    """
    Prints, for each dataset sequence evaluated by results_MG.py, the longest 3D distance between
    any two ground-truth positions across all of that sequence's robots (its extent). Then prints
    each dataset's mean extent across its sequences, and its RMS ATE failure threshold as
    ``failure_fraction`` of that mean extent.
    """
    mode = results_MG.GroupingMode.GLOBAL
    robot_groups = results_MG._make_airmuseum_groups(mode) + results_MG._make_hercules_groups(mode) \
                   + results_MG._make_kimera_multi_groups(mode)
    failure_fraction = 0.10

    extents_by_dataset: Dict[str, List[float]] = {}
    print(f"{'Dataset':<14}{'Sequence':<34}{'Label':<20}{'Max Distance (m)':>18}")
    for group in robot_groups:
        gt_data_lst = results_MG._LOAD_GT_DATA_BY_DATASET_NAME[group.dataset_name](group.dataset_seq, list(group.robots))
        points: np.ndarray = np.concatenate([gt.positions.astype(float) for gt in gt_data_lst], axis=0)
        extent: float = max_pairwise_distance(points)
        extents_by_dataset.setdefault(group.dataset_name, []).append(extent)
        seq_label: str = group.label.split("\n")[-1]
        print(f"{group.dataset_name:<14}{group.dataset_seq:<34}{seq_label:<20}{extent:>18.2f}")

    print(f"\n{'Dataset':<14}{'Mean Extent (m)':>16}{f'Failure Threshold (m, {failure_fraction:.0%})':>32}")
    for dataset_name, extents in extents_by_dataset.items():
        mean_extent: float = float(np.mean(extents))
        print(f"{dataset_name:<14}{mean_extent:>16.2f}{failure_fraction * mean_extent:>32.2f}")

if __name__ == "__main__":
    main()

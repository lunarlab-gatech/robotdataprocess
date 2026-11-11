import copy
from decimal import Decimal
import numpy as np
from pathlib import Path
from robotdataprocess import OdometryData
from robotdataprocess.utils.conversion_utils import dec_arr_to_float_arr
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "results"))
import results_KimeraVIO
import results_ROMAN

def rms_ate(gt_data: OdometryData, est_data: OdometryData) -> float:
    """RMS ATE (m) of est_data against gt_data, using results_KimeraVIO's evaluation settings."""
    metrics, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    return metrics.APE.translation_part.rmse

def main():
    """
    Computes the precomputed Kimera-VIO RMS ATE for every tunnels robot, to check whether the two
    robots whose camera clock is ahead of the GT clock (sparkal1, sparkal2) have meaningfully worse
    ATE than the rest. Also recomputes their ATE with the GT shifted by the measured offset, since a
    within-robot drop isolates the offset's effect from differences in path difficulty.
    """
    dataset_seq = results_ROMAN.TUNNELS_SEQ
    gt_shifts = {"sparkal1": Decimal('4.40'), "sparkal2": Decimal('1.05')}  # from verify_vio_gt_time_offset.py

    rows = []
    for robot in results_ROMAN.TUNNELS_ROBOTS:
        est_data, gt_data = results_KimeraVIO.load_vio_and_gt(dataset_seq, robot)
        path_len: float = float(np.linalg.norm(np.diff(dec_arr_to_float_arr(gt_data.positions), axis=0), axis=1).sum())
        ate: float = rms_ate(gt_data, est_data)
        row = f"{robot:12s} {path_len:9.1f} {ate:9.2f} {100 * ate / path_len:8.2f}"

        # Recompute with GT shifted onto the camera clock
        if robot in gt_shifts:
            gt_shifted = copy.deepcopy(gt_data)
            gt_shifted.timestamps = gt_shifted.timestamps + gt_shifts[robot]
            ate_shifted: float = rms_ate(gt_shifted, est_data)
            row += f"  | +{gt_shifts[robot]} s {ate_shifted:9.2f} {100 * ate_shifted / path_len:8.2f}"
        rows.append(row)

    print(f"{'robot':12s} {'length m':>9s} {'ATE m':>9s} {'ATE %':>8s}  | {'GT shift':>8s} {'ATE m':>9s} {'ATE %':>8s}")
    print('\n'.join(rows))


if __name__ == "__main__":
    main()

import getpass
import numpy as np
from pathlib import Path
from robotdataprocess import CoordinateFrame, OdometryData, PathData
from robotdataprocess.utils.conversion_utils import dec_arr_to_float_arr
from scipy.spatial.transform import Rotation as R, Slerp
import sys
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "results"))
import results_ROMAN

GRID_DT = 0.05  # seconds between profile samples
WINDOW = 1.0  # seconds spanned by each speed estimate
OFFSETS = np.round(np.arange(-10.0, 10.0 + 1e-9, 0.05), 2)

def make_grid(t_start: float, t_end: float) -> np.ndarray:
    """Uniform time grid that leaves half a WINDOW of margin at each end."""
    return np.arange(t_start + WINDOW, t_end - WINDOW, GRID_DT)

def motion_profiles(path: PathData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frame-independent linear and angular speed of a trajectory, each the motion between
    t - WINDOW/2 and t + WINDOW/2 divided by WINDOW.

    Args:
        path (PathData): Trajectory to profile.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Grid times (s), linear speed (m/s), angular speed (rad/s).
    """
    t: np.ndarray = dec_arr_to_float_arr(path.timestamps)
    t, unique_idx = np.unique(t, return_index=True)
    p: np.ndarray = dec_arr_to_float_arr(path.positions)[unique_idx]
    q: np.ndarray = dec_arr_to_float_arr(path.orientations)[unique_idx]

    grid: np.ndarray = make_grid(t[0], t[-1])
    p_before = np.stack([np.interp(grid - WINDOW / 2, t, p[:, i]) for i in range(3)], axis=1)
    p_after = np.stack([np.interp(grid + WINDOW / 2, t, p[:, i]) for i in range(3)], axis=1)
    slerp = Slerp(t, R.from_quat(q))
    ang_speed: np.ndarray = (slerp(grid - WINDOW / 2).inv() * slerp(grid + WINDOW / 2)).magnitude() / WINDOW
    return grid, np.linalg.norm(p_after - p_before, axis=1) / WINDOW, ang_speed

def cross_correlate(grid_a: np.ndarray, a: np.ndarray, grid_b: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of a(t) with b(t + offset) for every offset in OFFSETS (NaN if overlap < 10 s)."""
    corrs = np.full(len(OFFSETS), np.nan)
    for i, offset in enumerate(OFFSETS):
        b_shifted = np.interp(grid_a + offset, grid_b, b, left=np.nan, right=np.nan)
        valid = ~np.isnan(b_shifted)
        if valid.sum() * GRID_DT >= 10.0:
            corrs[i] = np.corrcoef(a[valid], b_shifted[valid])[0, 1]
    return corrs

def summarize(corrs: np.ndarray) -> str:
    """Formats correlation at zero offset, the best correlation, and the best offset."""
    best = int(np.nanargmax(corrs))
    return f"{corrs[OFFSETS == 0.0][0]:6.3f} {corrs[best]:6.3f} {OFFSETS[best]:+6.2f}"

def main():
    """
    Estimates, for every Kimera-Multi robot-sequence, the time offset between the precomputed
    Kimera-VIO odometry (header stamps) and the ground-truth CSV by cross-correlating their
    linear and angular speed profiles. A positive offset means VIO time = GT time + offset.
    """
    user = getpass.getuser()
    data_dir = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data'
    seqs_and_robots = [
        (results_ROMAN.TUNNELS_SEQ, results_ROMAN.TUNNELS_ROBOTS),
        (results_ROMAN.HYBRID_SEQ, results_ROMAN.HYBRID_ROBOTS),
        (results_ROMAN.OUTDOOR_SEQ, results_ROMAN.OUTDOOR_ROBOTS),
    ]

    rows = []
    for dataset_seq, robots in seqs_and_robots:
        gt_by_robot = dict(zip(robots, results_ROMAN.load_gt_data_ROMAN(dataset_seq, list(robots))))
        for robot in robots:
            vio = OdometryData.from_ros1_bag(data_dir / dataset_seq / 'Kimera-VIO-Odom' / (robot + '.bag'),
                                             '/' + robot + '/kimera_vio_ros/odometry', CoordinateFrame.NONE)
            gt_grid, gt_lin, gt_ang = motion_profiles(gt_by_robot[robot])
            vio_grid, vio_lin, vio_ang = motion_profiles(vio)
            start_diff = float(vio.timestamps[0] - gt_by_robot[robot].timestamps[0])
            rows.append(f"{dataset_seq.split('_')[1]:8s} {robot:12s} {start_diff:+8.2f} | "
                        f"{summarize(cross_correlate(gt_grid, gt_lin, vio_grid, vio_lin))} | "
                        f"{summarize(cross_correlate(gt_grid, gt_ang, vio_grid, vio_ang))}")

    print(f"{'seq':8s} {'robot':12s} {'dStart':>8s} | {'lin c0':>6s} {'best':>6s} {'offset':>6s} | {'ang c0':>6s} {'best':>6s} {'offset':>6s}")
    print('\n'.join(rows))


if __name__ == "__main__":
    main()

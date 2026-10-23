from check_vio_gt_time_offset import cross_correlate, motion_profiles, OFFSETS
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

GRID_DT = 0.5  # seconds between window starts
TURN_WINDOW = 2.0  # seconds spanned by each turn-axis rotation
HAND_EYE_WINDOW = 5.0  # seconds spanned by each GT/VIO relative-rotation pair
VEL_WINDOW = 1.0  # seconds spanned by each velocity estimate
MAX_GAP = 1.0  # max seconds between real samples around an interpolated time
MIN_TURN_DEG = 10.0  # windows rotating less than this are ignored for axis estimates
MIN_SPEED = 0.3  # m/s; slower velocity windows are ignored
T_CAMERA_FLU = np.array([[0, -1,  0, 0],
                         [0,  0, -1, 0],
                         [1,  0,  0, 0],
                         [0,  0,  0, 1]], dtype=np.float64)  # Nominal T_camera_flu from MeronomyGraph's kimera_multi.py

class Trajectory:
    """Float-valued, interpolable view of a PathData used for frame-local motion statistics."""

    t: np.ndarray
    p: np.ndarray
    slerp: Slerp

    def __init__(self, path: PathData, time_offset: float = 0.0):
        t: np.ndarray = dec_arr_to_float_arr(path.timestamps) - time_offset
        self.t, unique_idx = np.unique(t, return_index=True)
        self.p: np.ndarray = dec_arr_to_float_arr(path.positions)[unique_idx]
        self.slerp: Slerp = Slerp(self.t, R.from_quat(dec_arr_to_float_arr(path.orientations)[unique_idx]))

    def valid(self, times: np.ndarray) -> np.ndarray:
        """True where each time lies inside the trajectory and between real samples at most MAX_GAP apart."""
        idx = np.clip(np.searchsorted(self.t, times), 1, len(self.t) - 1)
        inside = (times >= self.t[0]) & (times <= self.t[-1])
        return inside & (self.t[idx] - self.t[idx - 1] <= MAX_GAP)

    def relative_rotations(self, starts: np.ndarray, window: float) -> R:
        """Body-frame rotation from each start time to start + window."""
        return self.slerp(starts).inv() * self.slerp(starts + window)

    def body_velocities(self, times: np.ndarray) -> np.ndarray:
        """Velocity (m/s) at each time over VEL_WINDOW, expressed in the body frame."""
        p_before = np.stack([np.interp(times - VEL_WINDOW / 2, self.t, self.p[:, i]) for i in range(3)], axis=1)
        p_after = np.stack([np.interp(times + VEL_WINDOW / 2, self.t, self.p[:, i]) for i in range(3)], axis=1)
        return self.slerp(times).inv().apply((p_after - p_before) / VEL_WINDOW)

def window_starts(trajs: Tuple[Trajectory, ...], window: float) -> np.ndarray:
    """Grid of window start times where every trajectory has data at both window ends."""
    t_start = max(traj.t[0] for traj in trajs)
    t_end = min(traj.t[-1] for traj in trajs) - window
    starts = np.arange(t_start, t_end, GRID_DT)
    for traj in trajs:
        starts = starts[traj.valid(starts) & traj.valid(starts + window)]
    return starts

def turn_axis(traj: Trajectory) -> np.ndarray:
    """
    Body-frame unit turn axis of a ground robot, which is the vehicle's up axis, summed over
    turning windows so larger turns weigh more. A body frame pitched nose-down by theta (about
    +y, FLU) sees it lean towards -x by theta.

    Args:
        traj (Trajectory): Trajectory whose body frame is (nominally) FLU.
    Returns:
        np.ndarray: Unit up axis in the body frame.
    """
    rotvecs: np.ndarray = traj.relative_rotations(window_starts((traj,), TURN_WINDOW), TURN_WINDOW).as_rotvec()
    rotvecs = rotvecs[np.linalg.norm(rotvecs, axis=1) > np.radians(MIN_TURN_DEG)]
    axis: np.ndarray = np.sum(rotvecs * np.sign(rotvecs[:, 2:3]), axis=0)
    return axis / np.linalg.norm(axis)

def forward_direction(traj: Trajectory) -> np.ndarray:
    """
    Body-frame unit direction of a ground robot's forward motion (component-wise median over
    forward-moving windows), which is the vehicle's +x axis. A body frame pitched nose-down by
    theta sees forward motion climb by theta.

    Args:
        traj (Trajectory): Trajectory whose body frame is (nominally) FLU.
    Returns:
        np.ndarray: Unit forward direction in the body frame.
    """
    times: np.ndarray = window_starts((traj,), VEL_WINDOW) + VEL_WINDOW / 2
    v: np.ndarray = traj.body_velocities(times)
    v = v[(np.linalg.norm(v, axis=1) > MIN_SPEED) & (v[:, 0] > 0)]
    v_median: np.ndarray = np.median(v / np.linalg.norm(v, axis=1, keepdims=True), axis=0)
    return v_median / np.linalg.norm(v_median)

def vehicle_rotation(traj: Trajectory) -> R:
    """Rotation from a trajectory's body frame to the vehicle frame (FLU), from its turn axis and forward direction."""
    vehicle_rot, _ = R.align_vectors([[0, 0, 1], [1, 0, 0]], [turn_axis(traj), forward_direction(traj)])
    return vehicle_rot

def hand_eye_rotation(gt: Trajectory, vio: Trajectory) -> Tuple[np.ndarray, float, int]:
    """
    Rotation X from the VIO body frame to the GT body frame such that each GT body-frame motion
    A = X B X^-1 for the simultaneous VIO motion B, i.e. rotvec(A) = X rotvec(B) (Kabsch, angle-weighted).

    Args:
        gt (Trajectory): Ground-truth trajectory.
        vio (Trajectory): VIO trajectory with its body frame converted to nominal FLU, time-aligned to gt.
    Returns:
        Tuple[np.ndarray, float, int]: X as a rotation vector (deg), median residual (deg), and number of pairs.
    """
    starts: np.ndarray = window_starts((gt, vio), HAND_EYE_WINDOW)
    a: np.ndarray = gt.relative_rotations(starts, HAND_EYE_WINDOW).as_rotvec()
    b: np.ndarray = vio.relative_rotations(starts, HAND_EYE_WINDOW).as_rotvec()
    keep: np.ndarray = np.linalg.norm(a, axis=1) > np.radians(MIN_TURN_DEG)
    a, b = a[keep], b[keep]
    X, _ = R.align_vectors(a, b)
    residuals: np.ndarray = np.degrees(np.linalg.norm(a - X.apply(b), axis=1))
    return np.degrees(X.as_rotvec()), float(np.median(residuals)), len(a)

def best_time_offset(gt: PathData, vio: PathData) -> float:
    """VIO time minus GT time (s) maximizing linear-speed correlation; see check_vio_gt_time_offset.py."""
    gt_grid, gt_lin, _ = motion_profiles(gt)
    vio_grid, vio_lin, _ = motion_profiles(vio)
    return float(OFFSETS[int(np.nanargmax(cross_correlate(gt_grid, gt_lin, vio_grid, vio_lin)))])

def stream_stats(traj: Trajectory) -> str:
    """Formats turn-axis and velocity pitch-down estimates for one stream."""
    up: np.ndarray = turn_axis(traj)
    fwd: np.ndarray = forward_direction(traj)
    return (f"{np.degrees(np.arctan2(-up[0], up[2])):+5.1f} {np.degrees(np.arctan2(up[1], up[2])):+5.1f} "
            f"{np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))):+5.1f} {np.degrees(np.arctan2(fwd[1], fwd[0])):+5.1f}")

def main():
    """
    Locates the few-degree rotation between the Kimera-Multi GT body frame and the precomputed
    Kimera-VIO body frame (after the nominal T_camera_flu), for every robot-sequence. For the GT
    CSV and the VIO bag each on its own, it measures the body-frame tilt of the turn axis and of
    the forward velocity, which for a ground robot reveal how that body frame is pitched/rolled
    relative to the vehicle, and combines the two streams into a GT body -> camera roll/pitch/yaw.
    It then estimates the GT<-VIO body rotation directly by hand-eye
    calibration on time-aligned relative rotations; its z (yaw) component is poorly observable
    because ground robots turn almost only about z.
    """
    user = getpass.getuser()
    data_dir = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data'
    seqs_and_robots = [
        (results_ROMAN.TUNNELS_SEQ, results_ROMAN.TUNNELS_ROBOTS),
        (results_ROMAN.HYBRID_SEQ, results_ROMAN.HYBRID_ROBOTS),
        (results_ROMAN.OUTDOOR_SEQ, results_ROMAN.OUTDOOR_ROBOTS),
    ]

    stream_rows = []
    camera_rows = []
    hand_eye_rows = []
    for dataset_seq, robots in seqs_and_robots:
        gt_by_robot = dict(zip(robots, results_ROMAN.load_gt_data_ROMAN(dataset_seq, list(robots))))
        for robot in robots:
            label: str = f"{dataset_seq.split('_')[1]:8s} {robot:12s}"
            vio: OdometryData = OdometryData.from_ros1_bag(data_dir / dataset_seq / 'Kimera-VIO-Odom' / (robot + '.bag'),
                                             '/' + robot + '/kimera_vio_ros/odometry', CoordinateFrame.NONE)
            vio.apply_transformation_right_side(T_CAMERA_FLU)
            offset: float = best_time_offset(gt_by_robot[robot], vio)
            gt_traj = Trajectory(gt_by_robot[robot])
            vio_traj = Trajectory(vio, offset)
            stream_rows.append(f"{label} | {stream_stats(gt_traj)} | {stream_stats(vio_traj)}")

            yaw, pitch, roll = (vehicle_rotation(gt_traj).inv() * vehicle_rotation(vio_traj)).as_euler('ZYX', degrees=True)
            camera_rows.append(f"{label} | {roll:+5.1f} {pitch:+5.1f} {yaw:+5.1f}")

            rotvec, residual, num_pairs = hand_eye_rotation(gt_traj, vio_traj)
            hand_eye_rows.append(f"{label} {offset:+6.2f} | {np.linalg.norm(rotvec):5.1f} "
                                 f"[{rotvec[0]:+5.1f} {rotvec[1]:+5.1f} {rotvec[2]:+5.1f}] {residual:5.2f} {num_pairs:5d}")

    stream_header = "pDown  rollR vPDwn vHead"
    print("Per-stream body-frame tilt (deg; turn axis: pitch-down, roll-right; velocity: pitch-down, heading-left)")
    print(f"{'seq':8s} {'robot':12s} | {'GT ' + stream_header:>26s} | {'VIO ' + stream_header:>27s}")
    print('\n'.join(stream_rows))
    print("\nGT body -> camera (nominal FLU) rotation (deg; intrinsic ZYX; +roll right-side down, +pitch nose down, +yaw left)")
    print(f"{'seq':8s} {'robot':12s} | {'roll':>5s} {'pitch':>5s} {'yaw':>5s}")
    print('\n'.join(camera_rows))
    print("\nHand-eye GT <- VIO body rotation (deg), VIO time-aligned by offset (s)")
    print(f"{'seq':8s} {'robot':12s} {'offset':>6s} | {'angle':>5s} {'rotvec (x y z)':^19s} {'resid':>5s} {'pairs':>5s}")
    print('\n'.join(hand_eye_rows))


if __name__ == "__main__":
    main()

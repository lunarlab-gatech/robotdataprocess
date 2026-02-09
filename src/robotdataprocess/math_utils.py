from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interpolate_poses(ts: np.ndarray, pos: np.ndarray, quat: np.ndarray,
                      target_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate poses at target timestamps using linear interpolation for
    positions and SLERP for orientations.

    Args:
        ts: (N,) float64 array of source timestamps (must be sorted ascending).
        pos: (N, 3) float64 array of positions.
        quat: (N, 4) float64 array of quaternions in xyzw format.
        target_ts: (M,) float64 array of target timestamps. Must be within
            [ts[0], ts[-1]].

    Returns:
        new_pos: (M, 3) float64 interpolated positions.
        new_quat: (M, 4) float64 interpolated quaternions in xyzw format.

    Raises:
        ValueError: If target_ts is empty or any target timestamp is outside
            the source range.
    """
    if len(target_ts) == 0:
        raise ValueError("target_ts must not be empty.")

    if np.any(target_ts < ts[0]) or np.any(target_ts > ts[-1]):
        raise ValueError(
            f"Target timestamps [{target_ts.min()}, {target_ts.max()}] are outside "
            f"source range [{ts[0]}, {ts[-1]}]."
        )

    # Linear interpolation for position
    new_pos = np.zeros((len(target_ts), 3), dtype=np.float64)
    for i in range(3):
        new_pos[:, i] = np.interp(target_ts, ts, pos[:, i])

    # SLERP for orientation
    key_rots = R.from_quat(quat)
    slerp = Slerp(ts, key_rots)
    new_rots = slerp(target_ts)
    new_quat = new_rots.as_quat()  # returns xyzw by default

    return new_pos, new_quat

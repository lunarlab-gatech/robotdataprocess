from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Union


def interpolate_poses(ts: np.ndarray, pos: np.ndarray,
                      quat_or_slerp: Union[np.ndarray, Slerp],
                      target_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate poses at target timestamps using linear interpolation for
    positions and SLERP for orientations.

    Args:
        ts: (N,) float64 array of source timestamps (must be sorted ascending).
        pos: (N, 3) float64 array of positions.
        quat_or_slerp: Either an (N, 4) float64 array of quaternions in xyzw
            format, or a pre-built ``scipy.spatial.transform.Slerp`` object.
            Passing a pre-built ``Slerp`` avoids rebuilding it on every call,
            which significantly speeds up repeated queries against the same
            trajectory.
        target_ts: (M,) float64 array of target timestamps. Must be within
            [ts[0], ts[-1]].

    Returns:
        new_pos: (M, 3) float64 interpolated positions.
        new_quat: (M, 4) float64 interpolated quaternions in xyzw format.

    Raises:
        ValueError: If target_ts is empty, any target timestamp is outside
            the source range, or quat_or_slerp is not a valid type.
    """
    if len(target_ts) == 0:
        raise ValueError("target_ts must not be empty.")

    if np.any(target_ts < ts[0]) or np.any(target_ts > ts[-1]):
        raise ValueError(
            f"Target timestamps [{target_ts.min()}, {target_ts.max()}] are outside "
            f"source range [{ts[0]}, {ts[-1]}]."
        )

    if isinstance(quat_or_slerp, np.ndarray):
        slerp = Slerp(ts, R.from_quat(quat_or_slerp))
    elif isinstance(quat_or_slerp, Slerp):
        slerp = quat_or_slerp
    else:
        raise ValueError("quat_or_slerp must be an (N, 4) quaternion array or a Slerp object.")

    # Linear interpolation for position
    new_pos = np.zeros((len(target_ts), 3), dtype=np.float64)
    for i in range(3):
        new_pos[:, i] = np.interp(target_ts, ts, pos[:, i])

    # SLERP for orientation
    new_quat = slerp(target_ts).as_quat()  # returns xyzw by default

    return new_pos, new_quat

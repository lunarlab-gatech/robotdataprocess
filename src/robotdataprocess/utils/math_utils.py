from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import Sequence, Union


def nearest_index(sorted_ts: Union[np.ndarray, Sequence[float]],
                   t: Union[float, np.ndarray, Sequence[float]]) -> Union[int, np.ndarray]:
    """
    Index (or indices) of the entry/entries in a sorted timestamp array closest to
    query time(s).

    Snaps a time onto an existing sample grid -- e.g. anchoring a loop closure's
    timestamp onto the pose-graph vertices it must reference. A query outside the
    array's range clamps to the nearest end rather than raising, and a query exactly
    between two samples resolves to the earlier one.

    Args:
        sorted_ts: (N,) timestamps, sorted ascending.
        t: A single query timestamp, or an (M,) array of query timestamps.

    Returns:
        A python int for a scalar t, or an (M,) int array for an array t -- the
        index/indices into sorted_ts of the closest timestamp(s).

    Raises:
        ValueError: If sorted_ts is empty.
    """
    ts = np.asarray(sorted_ts, dtype=np.float64)
    if ts.size == 0:
        raise ValueError("sorted_ts must not be empty.")

    is_scalar = np.ndim(t) == 0
    queries = np.atleast_1d(np.asarray(t, dtype=np.float64))

    right_idx = np.clip(np.searchsorted(ts, queries, side='left'), 0, ts.size - 1)
    left_idx = np.clip(right_idx - 1, 0, ts.size - 1)
    use_right = np.abs(ts[right_idx] - queries) < np.abs(ts[left_idx] - queries)
    indices = np.where(use_right, right_idx, left_idx)

    return int(indices[0]) if is_scalar else indices


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

from __future__ import annotations

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from ..math_utils import interpolate_poses
from .Data import Data, CoordinateFrame
from decimal import Decimal
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
from typing import List, Tuple, Union


class LoopClosureData(Data):
    """
    Data class for inter-robot loop closure measurements. Each loop closure
    represents a relative pose between two robots at specific timestamps.
    """

    timestamps_a: NDArray[Decimal]  # Timestamps for the first robot
    timestamps_b: NDArray[Decimal]  # Timestamps for the second robot
    names: list[tuple[str, str]]    # Robot name pairs per loop closure
    translations: NDArray[Decimal]  # (N, 3) translation vectors
    orientations: NDArray[Decimal]  # (N, 4) quaternions in xyzw format
    num_loop_closures: int

    @typechecked
    def __init__(self, timestamps_a: Union[np.ndarray, list],
                 timestamps_b: Union[np.ndarray, list],
                 names: List[Tuple[str, str]],
                 translations: Union[np.ndarray, list],
                 orientations: Union[np.ndarray, list]):
        super().__init__(frame_id="")
        self.timestamps_a = col_to_dec_arr(timestamps_a)
        self.timestamps_b = col_to_dec_arr(timestamps_b)
        self.names = names
        self.translations = col_to_dec_arr(translations)
        self.orientations = col_to_dec_arr(orientations)
        self.num_loop_closures = len(self.timestamps_a)

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_json(cls, json_path: Union[Path, str]) -> LoopClosureData:
        """
        Creates a LoopClosureData instance from a JSON file containing loop
        closure alignment data.

        Args:
            json_path: Path to the JSON file.

        Returns:
            LoopClosureData instance.
        """
        with open(str(json_path), 'r') as f:
            data = json.load(f)

        timestamps_a = []
        timestamps_b = []
        names = []
        translations = []
        orientations = []

        for entry in data:
            # Convert seconds + nanoseconds to Decimal timestamp
            ts_a = Decimal(str(entry["seconds"][0])) + Decimal(str(entry["nanoseconds"][0])) / Decimal("1000000000")
            ts_b = Decimal(str(entry["seconds"][1])) + Decimal(str(entry["nanoseconds"][1])) / Decimal("1000000000")
            timestamps_a.append(ts_a)
            timestamps_b.append(ts_b)

            names.append((entry["names"][0], entry["names"][1]))
            translations.append(entry["translation"])
            orientations.append(entry["rotation"])

        return cls(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
        )

    # =========================================================================
    # ============================ Error Methods ==============================
    # =========================================================================

    def calculate_errors(self, name_to_path: dict) -> dict:
        """
        Calculate translation and rotation errors for each loop closure by
        comparing the estimated relative transform against ground truth computed
        from PathData trajectories.

        The estimated loop closure provides the pose of the second robot (names[1])
        with respect to the first robot (names[0]). The GT relative transform is
        computed as T_A^{-1} * T_B, where A is names[0] and B is names[1].

        Args:
            name_to_path: Dict mapping robot names to their ground truth PathData.

        Returns:
            Dict with:
                "translation_errors": (N,) float64 array of translation magnitude errors in meters.
                "rotation_errors": (N,) float64 array of rotation angle errors in degrees.
        """
        from .PathData import PathData

        translation_errors = np.zeros(self.num_loop_closures, dtype=np.float64)
        rotation_errors = np.zeros(self.num_loop_closures, dtype=np.float64)

        # Group loop closures by (name_a, name_b) for efficient batch interpolation
        # First, collect all unique path data and their target timestamps
        path_interp_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for i in range(self.num_loop_closures):
            name_a, name_b = self.names[i]

            # Get GT poses via interpolation
            pos_a, quat_a = self._get_interpolated_pose(name_a, self.timestamps_a[i], name_to_path, path_interp_cache)
            pos_b, quat_b = self._get_interpolated_pose(name_b, self.timestamps_b[i], name_to_path, path_interp_cache)

            # Compute GT relative transform: T_A^{-1} * T_B
            R_a = R.from_quat(quat_a)
            R_b = R.from_quat(quat_b)
            R_rel_gt = R_a.inv() * R_b
            t_rel_gt = R_a.inv().apply(pos_b - pos_a)

            # Estimated relative transform from this loop closure
            t_est = dec_arr_to_float_arr(self.translations[i])
            R_est = R.from_quat(dec_arr_to_float_arr(self.orientations[i]))

            # Translation error: magnitude of difference
            translation_errors[i] = np.linalg.norm(t_est - t_rel_gt)

            # Rotation error: angle of R_gt^{-1} * R_est
            R_diff = R_rel_gt.inv() * R_est
            rotation_errors[i] = np.degrees(R_diff.magnitude())

        return {
            "translation_errors": translation_errors,
            "rotation_errors": rotation_errors,
        }

    def _get_interpolated_pose(self, name: str, timestamp: Decimal,
                               name_to_path: dict,
                               cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Get an interpolated pose for a robot at a specific timestamp.
        Uses a cache to avoid redundant conversion of PathData to float arrays.

        Returns:
            pos: (3,) float64 position
            quat: (4,) float64 quaternion in xyzw
        """
        if name not in name_to_path or name_to_path[name] is None:
            raise ValueError(f"Robot name '{name}' not found in name_to_path dict.")

        # Cache the float arrays for each path
        if name not in cache:
            path = name_to_path[name]
            ts_float = dec_arr_to_float_arr(path.timestamps)
            pos_float = dec_arr_to_float_arr(path.positions)
            quat_float = dec_arr_to_float_arr(path.orientations)
            cache[name] = (ts_float, pos_float, quat_float)

        ts_float, pos_float, quat_float = cache[name]
        target = np.array([float(timestamp)], dtype=np.float64)

        new_pos, new_quat = interpolate_poses(ts_float, pos_float, quat_float, target)
        return new_pos[0], new_quat[0]

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    @staticmethod
    def visualize_errors(errors: dict, show_plots: bool = True):
        """
        Plot histograms of translation and rotation errors.

        Args:
            errors: Dict from calculate_errors with "translation_errors" and
                "rotation_errors" keys.
            show_plots: If True, display plots. Set to False for testing.
        """
        trans_err = errors["translation_errors"]
        rot_err = errors["rotation_errors"]

        # Translation error histogram
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(trans_err, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_title("Loop Closure Translation Error Distribution")
        ax1.set_xlabel("Translation Error (m)")
        ax1.set_ylabel("Count")
        ax1.grid(True)
        fig1.tight_layout()

        # Rotation error histogram
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(rot_err, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_title("Loop Closure Rotation Error Distribution")
        ax2.set_xlabel("Rotation Error (degrees)")
        ax2.set_ylabel("Count")
        ax2.grid(True)
        fig2.tight_layout()

        if show_plots:
            plt.show()

        return fig1, fig2

    @staticmethod
    def visualize_success_rate(errors: dict, num_thresholds: int = 100,
                               show_plots: bool = True):
        """
        Plot the percentage of successful loop closures as a function of
        error threshold. Produces two separate plots: one for translation
        error and one for rotation error.

        Args:
            errors: Dict from calculate_errors.
            num_thresholds: Number of threshold values to evaluate.
            show_plots: If True, display plots. Set to False for testing.
        """
        trans_err = errors["translation_errors"]
        rot_err = errors["rotation_errors"]
        n = len(trans_err)

        # Translation success rate
        trans_thresholds = np.linspace(0, np.max(trans_err) * 1.1, num_thresholds)
        trans_success = np.array([np.sum(trans_err <= t) / n * 100 for t in trans_thresholds])

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(trans_thresholds, trans_success, color='blue', linewidth=2)
        ax1.set_title("Loop Closure Success Rate vs Translation Threshold")
        ax1.set_xlabel("Translation Threshold (m)")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_ylim(0, 105)
        ax1.grid(True)
        fig1.tight_layout()

        # Rotation success rate
        rot_thresholds = np.linspace(0, np.max(rot_err) * 1.1, num_thresholds)
        rot_success = np.array([np.sum(rot_err <= t) / n * 100 for t in rot_thresholds])

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(rot_thresholds, rot_success, color='orange', linewidth=2)
        ax2.set_title("Loop Closure Success Rate vs Rotation Threshold")
        ax2.set_xlabel("Rotation Threshold (degrees)")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True)
        fig2.tight_layout()

        if show_plots:
            plt.show()

        return fig1, fig2

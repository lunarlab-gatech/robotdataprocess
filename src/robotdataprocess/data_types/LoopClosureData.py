from __future__ import annotations
from xml.parsers.expat import errors

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from ..math_utils import interpolate_poses
from .Data import Data, CoordinateFrame
from decimal import Decimal
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from .PathData import PathData
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from typeguard import typechecked
from typing import List, Tuple, Union


class LoopClosureData(Data):
    """
    Data class for inter-robot loop closure measurements. Each loop closure
    represents a relative pose between two robots at specific timestamps.
    """

    timestamps_a: np.ndarray  # Timestamps for the first robot
    timestamps_b: np.ndarray  # Timestamps for the second robot
    names: list[tuple[str, str]]    # Robot name pairs per loop closure
    translations: np.ndarray  # (N, 3) translation vectors
    orientations: np.ndarray  # (N, 4) quaternions in xyzw format
    num_loop_closures: int

    @typechecked
    def __init__(self, timestamps_a: Union[np.ndarray, list], timestamps_b: Union[np.ndarray, list],
                 names: List[Tuple[str, str]], translations: Union[np.ndarray, list], orientations: Union[np.ndarray, list]):
                 
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
        with respect to the first robot (names[0]), or H_A->B. The GT relative transform is
        computed as (T_W->A)^{-1} * T_W->B, where A is names[0] and B is names[1].

        Args:
            name_to_path: Dict mapping robot names to their ground truth PathData.

        Returns:
            Dict with:
                "translation_errors": (N,) float64 array of translation magnitude errors in meters.
                "rotation_errors": (N,) float64 array of rotation angle errors in degrees.
        """

        # Arrays to hold the results
        translation_errors = np.zeros(self.num_loop_closures, dtype=np.float64)
        rotation_errors = np.zeros(self.num_loop_closures, dtype=np.float64)

        # Calculate error for each loop closure
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
            pathData: PathData = name_to_path[name]
            ts_float = dec_arr_to_float_arr(pathData.timestamps)
            pos_float = dec_arr_to_float_arr(pathData.positions)
            quat_float = dec_arr_to_float_arr(pathData.orientations)
            cache[name] = (ts_float, pos_float, quat_float)

        ts_float, pos_float, quat_float = cache[name]
        target = np.array([float(timestamp)], dtype=np.float64)

        new_pos, new_quat = interpolate_poses(ts_float, pos_float, quat_float, target)
        return new_pos[0], new_quat[0]

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    @staticmethod
    def visualize_errors(errors: List[dict], labels: List[str], show_plots: bool = True, bins: int = 60,) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot histograms of translation and rotation errors.

        Args:
            errors: List of dicts from calculate_errors with "translation_errors" and
                "rotation_errors" keys.
            show_plots: If True, display plots. Set to False for testing.
        """

        if len(labels) != len(errors):
            raise ValueError("Labels must have the same length as errors!")

        sns.set_theme(
            style="whitegrid",
            context="talk",
            palette="tab10",
        )

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Collect all data for shared bins
        all_trans = np.concatenate(
            [np.asarray(e["translation_errors"]) for e in errors]
        )
        all_rot = np.concatenate(
            [np.asarray(e["rotation_errors"]) for e in errors]
        )

        trans_bins = np.histogram_bin_edges(all_trans, bins=bins)
        rot_bins = np.histogram_bin_edges(all_rot, bins=bins)

        palette = sns.color_palette("tab10", len(errors))

        for err, label, color in zip(errors, labels, palette):
            sns.histplot(
                err["translation_errors"],
                bins=trans_bins,
                stat="count",
                alpha=0.45,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                ax=ax1,
                label=label,
            )

            sns.histplot(
                err["rotation_errors"],
                bins=rot_bins,
                stat="count",
                alpha=0.45,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                ax=ax2,
                label=label,
            )

        # Translation plot formatting
        ax1.set_title("Loop Closure Translation Error Distribution")
        ax1.set_xlabel("Translation Error (m)")
        ax1.set_ylabel("Count")
        ax1.legend(title="Run")
        sns.despine(ax=ax1)

        # Rotation plot formatting
        ax2.set_title("Loop Closure Rotation Error Distribution")
        ax2.set_xlabel("Rotation Error (degrees)")
        ax2.set_ylabel("Count")
        ax2.legend(title="Run")
        sns.despine(ax=ax2)

        fig1.tight_layout()
        fig2.tight_layout()

        if show_plots:
            plt.show()

        return fig1, fig2

    @staticmethod
    def visualize_success_rate(
        errors: List[dict],
        labels: List[str],
        num_thresholds: int = 100,
        show_plots: bool = True,
        max_translation_frac: float = 1.0,
        max_rotation_frac: float = 1.0,
    ):
        """
        Plot loop closure success as a function of error threshold.
        Produces six plots:
        1) Translation success rate (%)
        2) Rotation success rate (%)
        3) Translation count under threshold
        4) Rotation count under threshold
        5) Combined success rate (%)
        6) Combined count under threshold
        """

        if len(labels) != len(errors):
            raise ValueError("labels must have the same length as errors")

        if not (0 < max_translation_frac <= 1.0):
            raise ValueError("max_translation_frac must be in (0, 1]")

        if not (0 < max_rotation_frac <= 1.0):
            raise ValueError("max_rotation_frac must be in (0, 1]")

        sns.set_theme(
            style="whitegrid",
            context="talk",
            palette="tab10",
        )

        # Original four plots
        fig1, ax1 = plt.subplots(figsize=(10, 6))  # translation %
        fig2, ax2 = plt.subplots(figsize=(10, 6))  # rotation %
        fig3, ax3 = plt.subplots(figsize=(10, 6))  # translation count
        fig4, ax4 = plt.subplots(figsize=(10, 6))  # rotation count

        # Combined plots
        fig5, ax5 = plt.subplots(figsize=(10, 6))  # combined %
        fig6, ax6 = plt.subplots(figsize=(10, 6))  # combined count

        palette = sns.color_palette("tab10", len(errors))

        # Global maxima
        all_trans = np.concatenate([np.asarray(e["translation_errors"]) for e in errors])
        all_rot = np.concatenate([np.asarray(e["rotation_errors"]) for e in errors])

        max_trans = np.max(all_trans) * max_translation_frac
        max_rot = np.max(all_rot) * max_rotation_frac

        trans_thresholds = np.linspace(0, max_trans, num_thresholds)
        rot_thresholds = np.linspace(0, max_rot, num_thresholds)

        for err, label, color in zip(errors, labels, palette):
            trans_err = np.asarray(err["translation_errors"])
            rot_err = np.asarray(err["rotation_errors"])
            n = len(trans_err)

            trans_counts = np.array([np.sum(trans_err <= t) for t in trans_thresholds])
            rot_counts = np.array([np.sum(rot_err <= t) for t in rot_thresholds])

            trans_success = trans_counts / n * 100
            rot_success = rot_counts / n * 100

            # Original percentage plots
            ax1.plot(trans_thresholds, trans_success, linewidth=2.5, color=color, label=label)
            ax2.plot(rot_thresholds, rot_success, linewidth=2.5, color=color, label=label)

            # Original count plots
            ax3.plot(trans_thresholds, trans_counts, linewidth=2.5, color=color, label=label)
            ax4.plot(rot_thresholds, rot_counts, linewidth=2.5, color=color, label=label)

            # ---- Combined threshold calculations ----
            # 2D grid: every combination of translation and rotation thresholds
            combined_success = np.zeros((num_thresholds, num_thresholds))  # counts
            for i, t_thresh in enumerate(trans_thresholds):
                for j, r_thresh in enumerate(rot_thresholds):
                    combined_success[i, j] = np.sum((trans_err <= t_thresh) & (rot_err <= r_thresh))

            # Percentage
            combined_percent = combined_success / n * 100

            # For plotting, we will collapse the 2D grid to a line by showing diagonal
            # i.e., translation and rotation thresholds increasing together
            diag_idx = np.arange(min(num_thresholds, num_thresholds))
            diag_trans = trans_thresholds[diag_idx]
            diag_rot = rot_thresholds[diag_idx]
            diag_percent = combined_percent[diag_idx, diag_idx]
            diag_count = combined_success[diag_idx, diag_idx]

            # Combined plots
            ax5.plot(diag_trans, diag_percent, linewidth=2.5, color=color, label=label)
            ax6.plot(diag_trans, diag_count, linewidth=2.5, color=color, label=label)

        # ---- Formatting ----

        # Translation success %
        ax1.set_title("Loop Closure Success Rate vs Translation Threshold")
        ax1.set_xlabel("Translation Threshold (m)")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_ylim(0, 105)
        ax1.set_xlim(0, max_trans)
        ax1.legend(title="Run")
        sns.despine(ax=ax1)

        # Rotation success %
        ax2.set_title("Loop Closure Success Rate vs Rotation Threshold")
        ax2.set_xlabel("Rotation Threshold (degrees)")
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.set_xlim(0, max_rot)
        ax2.legend(title="Run")
        sns.despine(ax=ax2)

        # Translation count
        ax3.set_title("Loop Closures Under Translation Threshold")
        ax3.set_xlabel("Translation Threshold (m)")
        ax3.set_ylabel("Number of Loop Closures")
        ax3.set_xlim(0, max_trans)
        ax3.legend(title="Run")
        sns.despine(ax=ax3)

        # Rotation count
        ax4.set_title("Loop Closures Under Rotation Threshold")
        ax4.set_xlabel("Rotation Threshold (degrees)")
        ax4.set_ylabel("Number of Loop Closures")
        ax4.set_xlim(0, max_rot)
        ax4.legend(title="Run")
        sns.despine(ax=ax4)

        # Combined percentage
        ax5.set_title("Loop Closure Success Rate vs Combined Thresholds")
        ax5.set_xlabel("Translation Threshold (m)")
        ax5.set_ylabel("Success Rate (%)")
        ax5.set_ylim(0, 105)
        ax5.set_xlim(0, max_trans)
        ax5_top = ax5.twiny()
        ax5_top.set_xlim(0, max_rot)
        ax5_top.set_xlabel("Rotation Threshold (degrees)")
        ax5.legend(title="Run")
        sns.despine(ax=ax5, trim=True)

        # Combined count
        ax6.set_title("Loop Closures Under Combined Thresholds")
        ax6.set_xlabel("Translation Threshold (m)")
        ax6.set_ylabel("Number of Loop Closures")
        ax6.set_xlim(0, max_trans)
        ax6_top = ax6.twiny()
        ax6_top.set_xlim(0, max_rot)
        ax6_top.set_xlabel("Rotation Threshold (degrees)")
        ax6.legend(title="Run")
        sns.despine(ax=ax6, trim=True)

        for fig in (fig1, fig2, fig3, fig4, fig5, fig6):
            fig.tight_layout()

        if show_plots:
            plt.show()

        return fig1, fig2, fig3, fig4, fig5, fig6
    
    @staticmethod
    def visualize_error_scatter(
        errors: List[dict],
        labels: List[str],
        show_plots: bool = True,
        max_translation_frac: float = 1.0,
        max_rotation_frac: float = 1.0,
    ):
        """
        Scatter plot of loop closure errors (log-log scale): each point is one loop closure.
        X-axis: translation error
        Y-axis: rotation error
        Each dict in errors gets a separate color.

        Axes automatically expand to include all points with a small margin.
        """

        if len(labels) != len(errors):
            raise ValueError("labels must have the same length as errors")

        if not (0 < max_translation_frac <= 1.0):
            raise ValueError("max_translation_frac must be in (0, 1]")

        if not (0 < max_rotation_frac <= 1.0):
            raise ValueError("max_rotation_frac must be in (0, 1]")

        sns.set_theme(style="whitegrid", context="talk", palette="tab10")
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("tab10", len(errors))

        # Collect all translation and rotation errors
        all_trans = np.concatenate([np.asarray(e["translation_errors"]) for e in errors])
        all_rot = np.concatenate([np.asarray(e["rotation_errors"]) for e in errors])

        # Apply max fraction
        all_trans = all_trans[all_trans <= np.max(all_trans) * max_translation_frac]
        all_rot = all_rot[all_rot <= np.max(all_rot) * max_rotation_frac]

        # Determine axis limits with a small margin (e.g., 5%)
        x_min = np.min(all_trans) * 0.95
        x_max = np.max(all_trans) * 1.05
        y_min = np.min(all_rot) * 0.95
        y_max = np.max(all_rot) * 1.05

        for err, label, color in zip(errors, labels, palette):
            trans_err = np.asarray(err["translation_errors"])
            rot_err = np.asarray(err["rotation_errors"])

            # Mask points beyond the max fraction
            mask = (trans_err <= x_max) & (rot_err <= y_max)
            ax.scatter(
                trans_err[mask],
                rot_err[mask],
                alpha=0.8,
                s=200,            # bigger Xs
                color=color,
                label=label,
                marker='x',
                edgecolors='none',
                clip_on=False
            )

        ax.set_title("Loop Closure Errors Scatter Plot (Log-Log)")
        ax.set_xlabel("Translation Error (m)")
        ax.set_ylabel("Rotation Error (degrees)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(title="Run")
        sns.despine(ax=ax)
        fig.tight_layout()

        if show_plots:
            plt.show()

        return fig





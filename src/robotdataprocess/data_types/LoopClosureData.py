from __future__ import annotations

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from ..math_utils import interpolate_poses
from .Data import Data
from decimal import Decimal
import json
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, StrMethodFormatter
import numpy as np
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
    detected_inliers: np.ndarray # (N,) boolean array indicating inlier loop closures

    @typechecked
    def __init__(self, timestamps_a: Union[np.ndarray, list], timestamps_b: Union[np.ndarray, list],
                 names: List[Tuple[str, str]], translations: Union[np.ndarray, list], orientations: Union[np.ndarray, list],
                 detected_inliers: Union[np.ndarray, list, None] = None):
                 
        super().__init__(frame_id="")
        self.timestamps_a = col_to_dec_arr(timestamps_a)
        self.timestamps_b = col_to_dec_arr(timestamps_b)
        self.names = names
        self.translations = col_to_dec_arr(translations)
        self.orientations = col_to_dec_arr(orientations)
        self.num_loop_closures = len(self.timestamps_a)
        if detected_inliers is not None:
            self.detected_inliers = np.array(detected_inliers, dtype=bool)

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
            detected_inliers=None,
        )

    @classmethod
    def from_g2o(cls, g2o_path: Union[Path, str], time_path: Union[Path, str],
                 names_override: Union[tuple[str, str], None] = None) -> LoopClosureData:
        """
        Creates a LoopClosureData instance from a g2o file containing
        EDGE_SE3:QUAT entries, using a timestamp file to map keyframe
        indices to real timestamps.

        GTSAM symbol keys are decoded as (character << 56 | index). The
        character is converted to a robot id ('a' -> 0, 'b' -> 1, etc.)
        and used together with the index to look up the timestamp from
        the time file.

        The g2o quaternion order is (qx, qy, qz, qw), which matches the
        xyzw convention used by this class.

        Args:
            g2o_path: Path to the .g2o file.
            time_path: Path to the timestamp file. Each line has:
                robot_id keyframe_id timestamp_ns [ignored...]
            names_override: If passed, this tuple of (name_a, name_b) will
                be used for all loop closures instead of decoding from keys.

        Returns:
            LoopClosureData instance.
        """
        # Build lookup: (robot_id, keyframe_id) -> timestamp in seconds
        time_lookup: dict[tuple[int, int], Decimal] = {}
        with open(str(time_path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                robot_id = int(parts[0])
                keyframe_id = int(parts[1])
                timestamp_ns = Decimal(parts[2])
                time_lookup[(robot_id, keyframe_id)] = timestamp_ns / Decimal("1000000000")

        timestamps_a = []
        timestamps_b = []
        names = []
        translations = []
        orientations = []

        with open(str(g2o_path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("EDGE_SE3:QUAT"):
                    raise ValueError(
                        f"Expected EDGE_SE3:QUAT but got: {line.split()[0]}"
                    )

                parts = line.split()
                # parts[0]    = "EDGE_SE3:QUAT"
                # parts[1]    = key1
                # parts[2]    = key2
                # parts[3:6]  = px, py, pz
                # parts[6:10] = qx, qy, qz, qw
                # parts[10:]  = upper-triangular information matrix (21 values)

                key1 = int(parts[1])
                key2 = int(parts[2])

                # Decode GTSAM Symbol keys
                char1 = chr(key1 >> 56)
                idx1 = key1 & ((1 << 56) - 1)
                char2 = chr(key2 >> 56)
                idx2 = key2 & ((1 << 56) - 1)

                # Map character to robot index ('a' -> 0, 'b' -> 1, ...)
                robot_id1 = ord(char1) - ord('a')
                robot_id2 = ord(char2) - ord('a')

                timestamps_a.append(time_lookup[(robot_id1, idx1)])
                timestamps_b.append(time_lookup[(robot_id2, idx2)])
                if names_override is not None:
                    names.append(names_override)
                else:
                    names.append((char1, char2))

                px, py, pz = float(parts[3]), float(parts[4]), float(parts[5])
                qx, qy, qz, qw = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])

                translations.append([px, py, pz])
                orientations.append([qx, qy, qz, qw])

        return cls(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
            detected_inliers=None,
        )

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def round_timestamps(self, decimals: int):
        """
        Rounds all timestamps to the specified number of decimal places.

        Args:
            decimals: Number of decimal places to round to.
        """
        quantize_val = Decimal(10) ** -decimals
        self.timestamps_a = np.array([ts.quantize(quantize_val) for ts in self.timestamps_a])
        self.timestamps_b = np.array([ts.quantize(quantize_val) for ts in self.timestamps_b])

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
    # ===================== Multi LoopClosureData Methods =====================
    # =========================================================================

    def label_inliers_via_other_LoopClosureData(self, other: LoopClosureData) -> None:
        """
        Label loop closures as inliers if they are also detected in another
        LoopClosureData instance. The ``other`` object is assumed to be a
        subset of ``self`` — every loop closure in ``other`` must have a
        matching entry in ``self``. A ValueError is raised if any loop closure
        in ``other`` cannot be matched.

        This modifies the detected_inliers attribute in place.

        Matches are checked by name pairs, timestamps, translations, and
        orientations. Quaternion sign ambiguity is handled: q and -q are
        treated as equivalent.

        Note: Swapped name pairs ((A,B) vs (B,A)) are NOT matched. Both
        LoopClosureData instances must use the same robot1-to-robot2
        convention for loop closures to be identified as inliers.

        Args:
            other: Another LoopClosureData instance that is a subset of self;
                unaffected by this method.

        Raises:
            ValueError: If any loop closure in ``other`` is not found in
                ``self``, violating the subset assumption.
        """

        matched_other_indices = set()
        inliers = []

        for i in range(self.num_loop_closures):
            is_inlier = False
            for j in range(other.num_loop_closures):
                if (self.names[i] == other.names[j] and
                    self.timestamps_a[i] == other.timestamps_a[j] and
                    self.timestamps_b[i] == other.timestamps_b[j] and
                    np.allclose(dec_arr_to_float_arr(self.translations[i]),
                                dec_arr_to_float_arr(other.translations[j]),
                                atol=1e-4)):
                    q_self = dec_arr_to_float_arr(self.orientations[i])
                    q_other = dec_arr_to_float_arr(other.orientations[j])
                    if (np.allclose(q_self, q_other, atol=1e-4) or
                        np.allclose(q_self, -q_other, atol=1e-4)):
                        is_inlier = True
                        matched_other_indices.add(j)
                        break

            inliers.append(is_inlier)

        self.detected_inliers = np.array(inliers, dtype=bool)

        num_matched = int(np.sum(self.detected_inliers))
        if num_matched < other.num_loop_closures:
            raise ValueError(
                f"Only {num_matched} of {other.num_loop_closures} loop closures in other were found in self (other must be a subset of self)."
            )

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

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
        Produces six plots: translation success rate, rotation success rate,
        translation count, rotation count, combined success rate, and combined count.

        Args:
            errors: List of error dicts, each containing ``"translation_errors"``
                and ``"rotation_errors"`` arrays.
            labels: Display name for each error dict.
            num_thresholds: Number of evenly-spaced threshold values to evaluate.
            show_plots: If True, display the plots interactively.
            max_translation_frac: Fraction of the maximum translation error to
                use as the upper threshold bound.
            max_rotation_frac: Fraction of the maximum rotation error to
                use as the upper threshold bound.

        Returns:
            Tuple of six matplotlib Figure objects (translation %, rotation %,
            translation count, rotation count, combined %, combined count).

        Raises:
            ValueError: If list lengths do not match or fraction values are out of range.
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
        inlier_masks: List[np.ndarray] = None,
        show_plots: bool = True,
        save_path: str = None,
        max_translation_frac: float = 1.0,
        max_rotation_frac: float = 1.0,
        trans_err_in_target: float = 1.0,
        rot_err_in_target: float = 5.0,
        title: str = None
    ):
        """
        Scatter plot of loop closure errors (log-log scale): each point is one
        loop closure. Inliers and outliers are shown with different markers.

        Args:
            errors: List of error dicts, each containing ``"translation_errors"``
                and ``"rotation_errors"`` arrays.
            labels: Display name for each error dict.
            inlier_masks: Optional list of boolean arrays marking inlier loop
                closures. If None, all points are treated as outliers.
            show_plots: If True, display the plot interactively.
            save_path: If provided, save the figure to this path instead of showing.
            max_translation_frac: Fraction of max translation error for axis limit.
            max_rotation_frac: Fraction of max rotation error for axis limit.
            trans_err_in_target: Translation error threshold for the highlighted region.
            rot_err_in_target: Rotation error threshold for the highlighted region.
            title: Optional plot title.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: If list lengths do not match or fraction values are out of range.
            RuntimeError: If both ``show_plots`` and ``save_path`` are set.
        """

        if len(labels) != len(errors):
            raise ValueError("labels must have the same length as errors")

        if inlier_masks is not None and len(inlier_masks) != len(errors):
            raise ValueError("inlier_masks must have the same length as errors")

        if not (0 < max_translation_frac <= 1.0):
            raise ValueError("max_translation_frac must be in (0, 1]")

        if not (0 < max_rotation_frac <= 1.0):
            raise ValueError("max_rotation_frac must be in (0, 1]")

        if inlier_masks is None:
            inlier_masks = [None] * len(errors)

        sns.set_theme(style="whitegrid", context="talk", palette="tab10")
        sns.set_context("poster", font_scale=1.0)
        fig, ax = plt.subplots(figsize=(10, 10))
        palette = sns.color_palette("bright", len(errors))

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

        legend_handles = []
        for err, label, color, inlier_mask in zip(errors, labels, palette, inlier_masks):
            trans_err = np.asarray(err["translation_errors"])
            rot_err = np.asarray(err["rotation_errors"])

            # 1. Calculate how many points fall inside the highlighted square
            in_box_mask = (trans_err <= trans_err_in_target) & (rot_err <= rot_err_in_target)
            num_in_box = np.sum(in_box_mask)
            total_points = len(trans_err)

            # Avoid division by zero if a list is empty
            percent_in_box = (num_in_box / total_points * 100) if total_points > 0 else 0

            # Mask points beyond the max fraction for plotting
            vis_mask = (trans_err <= x_max) & (rot_err <= y_max)

            # Split into outliers (all points if no inlier_mask) and inliers
            if inlier_mask is not None:
                outlier_mask = vis_mask & ~inlier_mask
                inlier_vis_mask = vis_mask & inlier_mask
            else:
                outlier_mask = vis_mask
                inlier_vis_mask = np.zeros(len(trans_err), dtype=bool)

            # Calculate # of inliers
            if inlier_mask is not None:
                num_inliers = np.sum(inlier_mask)

            # Outliers
            ax.scatter(
                trans_err[outlier_mask], rot_err[outlier_mask],
                alpha=0.8, s=200, color=color,
                marker='x', edgecolors='none', clip_on=False, zorder=5,
            )
            # Inliers (same color, different marker, no extra legend entry)
            if np.any(inlier_vis_mask):
                ax.scatter(
                    trans_err[inlier_vis_mask], rot_err[inlier_vis_mask],
                    alpha=0.8, s=800, color=color,
                    marker='*', edgecolors='none', clip_on=False, zorder=5,
                )

            # Save the legend entry
            updated_label = f"{label}"
            # if inlier_mask is not None: updated_label += f" - Num Inliers: {num_inliers}"
            print(f"{label} ({percent_in_box:.1f}% ({num_in_box}/{total_points}) in target )")
            print(f"Number of inliers for {label}: {num_inliers}")
            legend_handles.append(Patch(facecolor=color, edgecolor='none', label=updated_label))
        
        if title is not None:
            ax.set_title(title, fontsize=24)
        ax.set_xlabel("Translation Error (m)")
        ax.set_ylabel("Rotation Error (degrees)")
        
        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Define exactly which "multipliers" get a tick and a label
        subs_to_show = [1.0, 2.0, 4.0, 6.0, 8.0]
        locator = LogLocator(base=10.0, subs=subs_to_show, numticks=100)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)

        # Use the 'g' formatter to ensure 0.1 shows as "0.1" and not "0"
        formatter = StrMethodFormatter('{x:g}')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        # Add the Highlighted Square
        import matplotlib.patches as patches
        rect = patches.Rectangle((1e-6, 1e-6), trans_err_in_target, rot_err_in_target, linewidth=0, facecolor='yellow', alpha=0.2, zorder=0)
        # legend_handles.append(Patch(facecolor='yellow', alpha=0.4, edgecolor='none', label='Successful Loop Closures (<=' + str(trans_err_in_target) + \
        #                             'm, <=' + str(rot_err_in_target) + '°)'))
        ax.add_patch(rect)

        # Place ticks at EVERY digit (1-9) so the grid lines exist
        all_subs = np.arange(1, 10) 
        locator = LogLocator(base=10.0, subs=all_subs, numticks=100)
        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(locator)

        # Use the 'g' formatter for clean decimals
        formatter = StrMethodFormatter('{x:g}')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        # Force a draw so Matplotlib generates the tick objects
        fig.canvas.draw()

        # Loop through and style: Grid thickness + Label visibility
        for axis in [ax.xaxis, ax.yaxis]:
            for tick in axis.get_major_ticks():
                val = tick.get_loc()
                log_val = np.log10(val)
                
                # Check if it's a power of 10 (1, 10, 100, 0.1, etc)
                is_power_of_10 = np.isclose(log_val, np.round(log_val), atol=1e-9)
                
                # Determine the "leading digit" (e.g., 0.02 -> 2, 40 -> 4)
                # This handles floating point math safely
                leading_digit = int(round(val / 10**np.floor(log_val + 1e-9)))

                # --- GRID STYLING ---
                if is_power_of_10:
                    tick.gridline.set_linewidth(2.5)
                    tick.gridline.set_color('#666666')
                    tick.gridline.set_alpha(0.8)
                else:
                    tick.gridline.set_linewidth(2.5)
                    tick.gridline.set_color("#CCCCCCD8")
                    tick.gridline.set_alpha(0.5)

        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')

        # --- LABEL VISIBILITY ---
        # Show labels for powers of 10 (1) and even numbers (2, 4, 6, 8)
        # Hide labels for odd numbers (3, 5, 7, 9)
        if leading_digit in [3, 5, 7, 9]:
            tick.label1.set_visible(False)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(title="Run", handles=legend_handles, frameon=True)
        sns.despine(ax=ax)
        fig.tight_layout()

        if show_plots and save_path is not None:
            raise RuntimeError("Can't enable both show_plots and save_path!")
        elif show_plots:
            plt.show()
        elif save_path is not None:
            plt.savefig(save_path)
        plt.close()

        return fig

from __future__ import annotations

from ...utils.conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from ...utils.math_utils import interpolate_poses
from scipy.spatial.transform import Slerp
from ..Data import Data
from .LoopClosureDataResult import LoopClosureDataResult
from collections import Counter
from decimal import Decimal
import itertools
import json
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, StrMethodFormatter
import numpy as np
from ..PathData import PathData
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from typeguard import typechecked
from typing import Dict, List, Optional, Tuple, Union

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
    results: Optional[LoopClosureDataResult]  # Set by calculate_errors(); None until then.

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
        self.results = None

    @staticmethod
    def _canonical_lc_key(name_pair: Tuple[str, str], ts_a: Decimal, ts_b: Decimal) -> Tuple[str, str, Decimal, Decimal]:
        """
        Builds a key identifying a loop closure by its name pair and timestamp
        pair, treating swapped pairs as identical -- i.e. ``(A, B, ts_a, ts_b)``
        and ``(B, A, ts_b, ts_a)`` produce the same key. The pair with the
        smaller ``(name, timestamp)`` tuple is always placed first.

        Args:
            name_pair: The ``(name_a, name_b)`` robot names for this loop closure.
            ts_a: Timestamp for ``name_a``.
            ts_b: Timestamp for ``name_b``.

        Returns:
            A canonical ``(name_a, name_b, ts_a, ts_b)`` tuple.
        """
        na, nb = name_pair
        return (na, nb, ts_a, ts_b) if (na, ts_a) <= (nb, ts_b) else (nb, na, ts_b, ts_a)

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_json(cls, json_path: Union[Path, str],
                  names_override: Union[dict, None] = None) -> LoopClosureData:
        """
        Creates a LoopClosureData instance from a JSON file containing loop
        closure alignment data, in the format written by :meth:`to_json`.

        The JSON file is a list of objects, one per loop closure, each with:
            - ``"seconds"``: ``[sec_a, sec_b]``, integer seconds for each
              robot's timestamp.
            - ``"nanoseconds"``: ``[ns_a, ns_b]``, integer nanoseconds for
              each robot's timestamp (added to ``"seconds"``).
            - ``"names"``: ``[name_a, name_b]``, robot names for this loop closure.
            - ``"translation"``: ``[tx, ty, tz]``, estimated relative translation.
            - ``"rotation"``: ``[qx, qy, qz, qw]``, estimated relative rotation quaternion.

        Args:
            json_path: Path to the JSON file.
            names_override: Optional dict mapping names found in the JSON to
                desired replacement names (e.g. ``{"0": "aerial-07",
                "1": "ground-03"}``). Names not present in the dict are kept
                as-is.

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

            if names_override is not None:
                name_a = names_override.get(entry["names"][0], entry["names"][0])
                name_b = names_override.get(entry["names"][1], entry["names"][1])
            else:
                name_a = entry["names"][0]
                name_b = entry["names"][1]
            names.append((name_a, name_b))
            translations.append(entry["translation"])
            orientations.append(entry["rotation"])

        return cls(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
        )

    @classmethod
    def from_g2o(cls, g2o_path: Union[Path, str], time_path: Union[Path, str],
                 names_override: Union[dict, None] = None) -> LoopClosureData:
        """
        Creates a LoopClosureData instance from a g2o file containing
        EDGE_SE3:QUAT loop closure entries, using a timestamp file to map
        keyframe indices to real timestamps.

        Loop closure edges must be preceded by a comment line starting with
        ``# LC:`` (e.g. ``# LC: some info``). Edges not preceded by ``# LC:``
        are treated as odometry and skipped. A sanity check raises ``ValueError``
        for any edge that lacks the ``# LC:`` marker but also cannot be an
        odometry edge (odometry requires the same robot character and
        consecutive keyframe indices, i.e. ``|idx1 - idx2| == 1``).

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
            names_override: Optional dict mapping decoded character keys to
                desired robot names (e.g. ``{"a": "aerial-07",
                "b": "ground-03"}``). Keys not present in the dict are kept
                as their decoded character.

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

        lc_marker_seen = False

        with open(str(g2o_path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('# LC:'):
                    lc_marker_seen = True
                    continue
                if line.startswith('#'):
                    continue
                if not line.startswith("EDGE_SE3:QUAT"):
                    lc_marker_seen = False
                    continue

                is_lc = lc_marker_seen
                lc_marker_seen = False

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

                could_be_odometry = (char1 == char2 and abs(idx1 - idx2) == 1)

                if not is_lc and not could_be_odometry:
                    raise ValueError(
                        f"Edge between robot '{char1}' keyframe {idx1} and robot '{char2}' "
                        f"keyframe {idx2} is not marked with '# LC:' but cannot be an odometry "
                        f"edge (odometry requires the same robot and consecutive keyframe indices)."
                    )

                if not is_lc:
                    continue  # odometry edge, skip

                # Map character to robot index ('a' -> 0, 'b' -> 1, ...)
                robot_id1 = ord(char1) - ord('a')
                robot_id2 = ord(char2) - ord('a')

                timestamps_a.append(time_lookup[(robot_id1, idx1)])
                timestamps_b.append(time_lookup[(robot_id2, idx2)])
                if names_override is not None:
                    name_a = names_override.get(char1, char1)
                    name_b = names_override.get(char2, char2)
                    names.append((name_a, name_b))
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
        )

    @classmethod
    def from_maplab_json(cls, json_path: Union[Path, str],
                         names_override: Union[dict, None] = None) -> LoopClosureData:
        """
        Creates a LoopClosureData instance from a maplab JSON file containing
        loop closure constraints.

        Each loop closure entry provides nanosecond timestamps, mission UUIDs,
        the relative transform ``T_A_B`` (translation and ``rotation_xyzw``),
        and a ``switch_variable`` flag (1 = inlier).

        Args:
            json_path: Path to the JSON file.
            names_override: Optional dict mapping mission UUIDs found in the
                JSON to desired replacement names (e.g.
                ``{"99a8765349fea5180b00000000000000": "robot-01"}``). UUIDs
                not present in the dict are kept as-is.

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

        for entry in data["loop_closures"]:
            ts_a = Decimal(str(entry["from_timestamp_ns"])) / Decimal("1000000000")
            ts_b = Decimal(str(entry["to_timestamp_ns"])) / Decimal("1000000000")
            timestamps_a.append(ts_a)
            timestamps_b.append(ts_b)

            name_a = entry["from_mission"]
            name_b = entry["to_mission"]
            if names_override is not None:
                name_a = names_override.get(name_a, name_a)
                name_b = names_override.get(name_b, name_b)
            names.append((name_a, name_b))

            translations.append(entry["T_A_B"]["translation"])
            orientations.append(entry["T_A_B"]["rotation_xyzw"])

        return cls(
            timestamps_a=np.array(timestamps_a, dtype=object),
            timestamps_b=np.array(timestamps_b, dtype=object),
            names=names,
            translations=np.array(translations, dtype=object),
            orientations=np.array(orientations, dtype=object),
        )

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def round_timestamps(self, decimals: int):
        """
        Rounds all timestamps to the specified number of decimal places.
        Invalidates ``self.results`` (sets it to None), since the rounding
        shifts the timestamps used to interpolate ground truth poses and
        could therefore change ``calculate_errors``' output.

        Args:
            decimals: Number of decimal places to round to.
        """
        quantize_val = Decimal(10) ** -decimals
        self.timestamps_a = np.array([ts.quantize(quantize_val) for ts in self.timestamps_a])
        self.timestamps_b = np.array([ts.quantize(quantize_val) for ts in self.timestamps_b])
        self.results = None

    def _prune_by_mask(self, mask: np.ndarray) -> None:
        self.timestamps_a = self.timestamps_a[mask]
        self.timestamps_b = self.timestamps_b[mask]
        self.names = [name for name, keep in zip(self.names, mask) if keep]
        self.translations = self.translations[mask]
        self.orientations = self.orientations[mask]
        if self.results is not None:
            self.results.translation_errors = self.results.translation_errors[mask]
            self.results.rotation_errors = self.results.rotation_errors[mask]
            if self.results.successful is not None:
                self.results.successful = self.results.successful[mask]
        self.num_loop_closures = len(self.timestamps_a)

    def prune_intra_robot_loop_closures(self):
        """
        Removes loop closures where both names in the pair are the same,
        i.e. intra-robot loop closures. Modifies the instance in place.
        """
        mask = np.array([name_a != name_b for name_a, name_b in self.names], dtype=bool)
        self._prune_by_mask(mask)

    def prune_inter_robot_loop_closures(self):
        """
        Removes loop closures where the two names in the pair differ,
        i.e. inter-robot loop closures. Modifies the instance in place.
        """
        mask = np.array([name_a == name_b for name_a, name_b in self.names], dtype=bool)
        self._prune_by_mask(mask)

    def _lc_transform_in_canonical_order(self, i: int, key: tuple) -> Tuple[np.ndarray, R]:
        t = dec_arr_to_float_arr(self.translations[i])
        rot = R.from_quat(dec_arr_to_float_arr(self.orientations[i]))
        if self.names[i][0] == key[0] and self.timestamps_a[i] == key[2]:
            return t, rot
        # This entry is the swapped half of the pair — invert so it's expressed
        # in the same direction as the canonical (non-swapped) entries.
        rot_inv = rot.inv()
        return rot_inv.apply(-t), rot_inv

    def prune_duplicates(self):
        """
        Removes duplicate loop closures, keeping only the first occurrence of
        each. Two loop closures are considered duplicates if they share the
        same name pair and timestamp pair, treating swapped pairs as identical
        — i.e. ``(A, B, ts_a, ts_b)`` and ``(B, A, ts_b, ts_a)`` are the same
        LC. Modifies the instance in place.
        """
        seen = set()
        mask = np.zeros(self.num_loop_closures, dtype=bool)
        for i in range(self.num_loop_closures):
            key = self._canonical_lc_key(self.names[i], self.timestamps_a[i], self.timestamps_b[i])
            if key not in seen:
                seen.add(key)
                mask[i] = True
        self._prune_by_mask(mask)

    # =========================================================================
    # ============================ Error Methods ==============================
    # =========================================================================

    def _get_interpolated_pose(self, name: str, timestamp: Decimal,
                                name_to_path: dict,
                                cache: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Get an interpolated pose for a robot at a specific timestamp.
        Uses a cache (keyed by robot name) to avoid redundant conversion of
        PathData to float arrays and to reuse pre-built Slerp objects across
        repeated calls for the same robot.

        Returns:
            pos: (3,) float64 position
            quat: (4,) float64 quaternion in xyzw
        """
        if name not in name_to_path or name_to_path[name] is None:
            raise ValueError(f"Robot name '{name}' not found in name_to_path dict.")

        # Cache float arrays and pre-built Slerp object per robot
        if name not in cache:
            pathData: PathData = name_to_path[name]
            ts_float = dec_arr_to_float_arr(pathData.timestamps)
            pos_float = dec_arr_to_float_arr(pathData.positions)
            quat_float = dec_arr_to_float_arr(pathData.orientations)
            slerp = Slerp(ts_float, R.from_quat(quat_float))
            cache[name] = (ts_float, pos_float, slerp)

        ts_float, pos_float, slerp = cache[name]

        # Clip the target timestamp (to be robust when baselines assume submaps start at t=0)
        target = np.clip(
            np.array([float(timestamp)], dtype=np.float64),
            ts_float[0], ts_float[-1],
        )

        new_pos, new_quat = interpolate_poses(ts_float, pos_float, slerp, target)
        return new_pos[0], new_quat[0]

    def calculate_errors(self, name_to_path: dict) -> None:
        """
        Calculate translation and rotation errors for each loop closure by
        comparing the estimated relative transform against ground truth computed
        from PathData trajectories. Sets ``self.results`` to a
        :class:`LoopClosureDataResult` holding the per-loop-closure errors.

        The estimated loop closure provides the pose of the second robot (names[1])
        with respect to the first robot (names[0]), or H_A->B. The GT relative transform is
        computed as (T_W->A)^{-1} * T_W->B, where A is names[0] and B is names[1].

        Args:
            name_to_path: Dict mapping robot names to their ground truth PathData.
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

        self.results = LoopClosureDataResult(translation_errors=translation_errors, rotation_errors=rotation_errors)

    def label_successful(self, trans_err_in_target: float, rot_err_in_target: float) -> None:
        """
        Labels each loop closure in ``self.results`` as successful. See
        :meth:`LoopClosureDataResult.label_successful`. ``calculate_errors``
        must be called first to populate ``self.results``.

        Raises:
            ValueError: If ``self.results`` is None, i.e. ``calculate_errors``
                has not been called yet.
        """
        if self.results is None:
            raise ValueError("self.results is None -- call calculate_errors() first.")
        self.results.label_successful(trans_err_in_target, rot_err_in_target)

    # =========================================================================
    # ========================= Verification Methods ==========================
    # =========================================================================

    def print_duplicate_info(self, label: str = "") -> None:
        """
        Print the number of duplicate loop closures in this instance. Two loop
        closures are considered duplicates if they share the same name pair and
        timestamp pair, treating swapped pairs as identical — i.e.
        ``(A, B, ts_a, ts_b)`` and ``(B, A, ts_b, ts_a)`` are the same LC. Also
        prints the average pairwise translation and rotation difference between
        the transformations of duplicate loop closures (swapped duplicates are
        inverted first so they're compared in the same direction).

        Args:
            label: Optional prefix printed before the stats (e.g. the dataset
                or run name) to distinguish output when called multiple times.
        """
        keys = [self._canonical_lc_key(self.names[i], self.timestamps_a[i], self.timestamps_b[i])
                for i in range(self.num_loop_closures)]
        counts = Counter(keys)
        num_dupes = sum(c - 1 for c in counts.values() if c > 1)
        num_unique = self.num_loop_closures - num_dupes

        groups: dict = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)

        translation_diffs = []
        rotation_diffs = []
        for key, idxs in groups.items():
            if len(idxs) < 2:
                continue
            transforms = [self._lc_transform_in_canonical_order(i, key) for i in idxs]
            for (t1, rot1), (t2, rot2) in itertools.combinations(transforms, 2):
                translation_diffs.append(np.linalg.norm(t1 - t2))
                rotation_diffs.append(np.degrees((rot1.inv() * rot2).magnitude()))

        prefix = f"{label}: " if label else ""
        msg = f"{prefix}{self.num_loop_closures} total loop closures, {num_dupes} duplicates ({num_unique} if deduplicated)"
        if translation_diffs:
            msg += (f", avg duplicate transform diff: {np.mean(translation_diffs):.4f} m, "
                    f"{np.mean(rotation_diffs):.4f} deg")
        print(msg)

    def print_successful_lc_diff(self, other: LoopClosureData, self_label: str = "self", other_label: str = "other") -> None:
        """
        For each loop closure labeled successful in this instance, checks whether
        a loop closure with the same name pair and timestamp pair (see
        :meth:`_canonical_lc_key`) exists in ``other``, and prints why it isn't
        also successful there: missing entirely, or present but over the
        translation/rotation error threshold.

        Both this instance and ``other`` must already have ``label_successful``
        called (i.e. ``self.results.successful`` populated).

        Args:
            other: The LoopClosureData instance to compare against.
            self_label: Label for this instance, used in printed messages.
            other_label: Label for ``other``, used in printed messages.

        Raises:
            ValueError: If either instance's ``self.results.successful`` is None.
        """
        if self.results is None or self.results.successful is None:
            raise ValueError(f"{self_label}: self.results.successful is None -- call label_successful() first.")
        if other.results is None or other.results.successful is None:
            raise ValueError(f"{other_label}: other.results.successful is None -- call label_successful() first.")

        # Build lookup: canonical (name_a, name_b, ts_a, ts_b) key -> other index
        lc_key_to_other_idx: Dict[Tuple[str, str, Decimal, Decimal], int] = {
            other._canonical_lc_key(other.names[i], other.timestamps_a[i], other.timestamps_b[i]): i
            for i in range(other.num_loop_closures)
        }

        # Iterate through successful LC in self
        for i in range(self.num_loop_closures):
            if not self.results.successful[i]:
                continue

            # Get canonical key and the corresponding LC in other
            key: Tuple[str, str, Decimal, Decimal] = self._canonical_lc_key(self.names[i], self.timestamps_a[i], self.timestamps_b[i])
            desc: str = f"{self.names[i][0]}@{self.timestamps_a[i]} <-> {self.names[i][1]}@{self.timestamps_b[i]}"
            match_idx: Optional[int] = lc_key_to_other_idx.get(key)

            # Detect if LC missing in other
            if match_idx is None:
                print(f"{desc}: successful in {self_label}, no corresponding LC found in {other_label}")
                continue

            # Detect if successful in both
            if other.results.successful[match_idx]:
                print(f"{desc}: successful in both {self_label} and {other_label}")
                continue

            trans_err: float = other.results.translation_errors[match_idx]
            rot_err: float = other.results.rotation_errors[match_idx]
            self_trans_err: float = self.results.translation_errors[i]
            self_rot_err: float = self.results.rotation_errors[i]
            reasons: List[str] = []
            if trans_err > other.results.trans_err_in_target:
                reasons.append(f"trans err {trans_err:.3f}m > {other.results.trans_err_in_target}m")
            if rot_err > other.results.rot_err_in_target:
                reasons.append(f"rot err {rot_err:.3f}deg > {other.results.rot_err_in_target}deg")
            print(f"{desc}: successful in {self_label}, found in {other_label} but not successful ({', '.join(reasons)}) "
                  f"({self_label} trans err {self_trans_err:.3f}m, rot err {self_rot_err:.3f}deg; "
                  f"{other_label} trans err {trans_err:.3f}m, rot err {rot_err:.3f}deg)")

    # =========================================================================
    # ============================ Export Methods ==============================
    # =========================================================================

    def to_json(self, json_path: Union[Path, str]) -> None:
        """
        Writes this LoopClosureData to a JSON file, in the format read by
        :meth:`from_json`.

        Args:
            json_path: Destination path for the JSON file.
        """
        data = []
        for i in range(self.num_loop_closures):
            seconds_a, remainder_a = divmod(self.timestamps_a[i], Decimal(1))
            seconds_b, remainder_b = divmod(self.timestamps_b[i], Decimal(1))
            nanoseconds_a = int(remainder_a * Decimal("1000000000"))
            nanoseconds_b = int(remainder_b * Decimal("1000000000"))
            name_a, name_b = self.names[i]

            data.append({
                "seconds": [int(seconds_a), int(seconds_b)],
                "nanoseconds": [nanoseconds_a, nanoseconds_b],
                "names": [name_a, name_b],
                "translation": dec_arr_to_float_arr(self.translations[i]).tolist(),
                "rotation": dec_arr_to_float_arr(self.orientations[i]).tolist(),
            })

        with open(str(json_path), 'w') as f:
            json.dump(data, f, indent=4)

    # =========================================================================
    # ===================== Multi LoopClosureData Methods =====================
    # =========================================================================

    @staticmethod
    def merge(loop_closures: List[LoopClosureData]) -> LoopClosureData:
        """
        Creates a new LoopClosureData containing all loop closures from each
        object in the list. The input objects and list are not modified.

        Args:
            loop_closures: List of LoopClosureData instances to merge.

        Returns:
            New LoopClosureData with all loop closures concatenated.
        """
        loop_closures = [lc for lc in loop_closures if lc.num_loop_closures > 0]
        if not loop_closures:
            return LoopClosureData([], [], [], [], [])
        timestamps_a = np.concatenate([lc.timestamps_a for lc in loop_closures])
        timestamps_b = np.concatenate([lc.timestamps_b for lc in loop_closures])
        names = [name for lc in loop_closures for name in lc.names]
        translations = np.concatenate([lc.translations for lc in loop_closures])
        orientations = np.concatenate([lc.orientations for lc in loop_closures])

        return LoopClosureData(timestamps_a, timestamps_b, names, translations, orientations)

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    @staticmethod
    def visualize_success_rate(
        lc_data_list: List[LoopClosureData],
        labels: List[str],
        num_thresholds: int = 100,
        show_plots: bool = True,
        max_translation_frac: float = 1.0,
        max_rotation_frac: float = 1.0,
        include_rate_plots: bool = True,
    ):
        """
        Plot loop closure success as a function of error threshold.

        When ``include_rate_plots`` is True (default), produces a figure with
        six subplots (3 rows × 2 columns): translation success rate, rotation
        success rate, translation count, rotation count, combined success rate,
        and combined count.

        When ``include_rate_plots`` is False, produces a figure with three
        subplots (1 row × 3 columns) showing only the count plots: translation
        count, rotation count, and combined count.

        Args:
            lc_data_list: LoopClosureData instances, each with ``calculate_errors``
                already called (i.e. ``.results`` populated).
            labels: Display name for each entry in ``lc_data_list``.
            num_thresholds: Number of evenly-spaced threshold values to evaluate.
            show_plots: If True, display the plots interactively.
            max_translation_frac: Fraction of the maximum translation error to
                use as the upper threshold bound.
            max_rotation_frac: Fraction of the maximum rotation error to
                use as the upper threshold bound.
            include_rate_plots: If False, omit the success-rate (%) subplots and
                show only the loop closure count subplots.

        Returns:
            A single matplotlib Figure.

        Raises:
            ValueError: If list lengths do not match or fraction values are out of range.
        """

        if len(labels) != len(lc_data_list):
            raise ValueError("labels must have the same length as lc_data_list")

        if not (0 < max_translation_frac <= 1.0):
            raise ValueError("max_translation_frac must be in (0, 1]")

        if not (0 < max_rotation_frac <= 1.0):
            raise ValueError("max_rotation_frac must be in (0, 1]")

        sns.set_theme(style="whitegrid", context="talk", palette="tab10")

        if include_rate_plots:
            fig, axes = plt.subplots(3, 2, figsize=(20, 18))
            ax1, ax2 = axes[0]
            ax3, ax4 = axes[1]
            ax5, ax6 = axes[2]
        else:
            fig, (ax3, ax4, ax6) = plt.subplots(1, 3, figsize=(30, 9))
            ax1 = ax2 = ax5 = None
        palette = sns.color_palette("tab10", len(lc_data_list))

        # Global maxima (fall back to a default range if there are no loop closures at all)
        all_trans = np.concatenate([np.asarray(lc.results.translation_errors) for lc in lc_data_list])
        all_rot = np.concatenate([np.asarray(lc.results.rotation_errors) for lc in lc_data_list])

        max_trans = np.max(all_trans) * max_translation_frac if all_trans.size else 1.0
        max_rot = np.max(all_rot) * max_rotation_frac if all_rot.size else 1.0

        trans_thresholds = np.linspace(0, max_trans, num_thresholds)
        rot_thresholds = np.linspace(0, max_rot, num_thresholds)

        for lc, label, color in zip(lc_data_list, labels, palette):
            trans_err = np.asarray(lc.results.translation_errors)
            rot_err = np.asarray(lc.results.rotation_errors)
            n = len(trans_err)

            trans_counts = np.array([np.sum(trans_err <= t) for t in trans_thresholds])
            rot_counts = np.array([np.sum(rot_err <= t) for t in rot_thresholds])

            if include_rate_plots:
                trans_success = trans_counts / n * 100 if n else np.zeros_like(trans_counts, dtype=float)
                rot_success = rot_counts / n * 100 if n else np.zeros_like(rot_counts, dtype=float)
                ax1.plot(trans_thresholds, trans_success, linewidth=2.5, color=color, label=label)
                ax2.plot(rot_thresholds, rot_success, linewidth=2.5, color=color, label=label)

            ax3.plot(trans_thresholds, trans_counts, linewidth=2.5, color=color, label=label)
            ax4.plot(rot_thresholds, rot_counts, linewidth=2.5, color=color, label=label)

            # Combined plots: 1 m translation = 5 deg rotation
            # Evaluate directly at (t, 5t) rather than re-indexing the grid.
            diag_trans = trans_thresholds
            diag_rot = 5.0 * diag_trans
            diag_count = np.array([np.sum((trans_err <= t) & (rot_err <= r))
                                   for t, r in zip(diag_trans, diag_rot)], dtype=float)

            if include_rate_plots:
                diag_percent = diag_count / n * 100 if n else np.zeros_like(diag_count, dtype=float)
                ax5.plot(diag_trans, diag_percent, linewidth=2.5, color=color, label=label)
            ax6.plot(diag_trans, diag_count, linewidth=2.5, color=color, label=label)

        # ---- Formatting ----

        if include_rate_plots:
            ax1.set_title("Loop Closure Success Rate vs Translation Threshold")
            ax1.set_xlabel("Translation Threshold (m)")
            ax1.set_ylabel("Success Rate (%)")
            ax1.set_ylim(0, 105)
            ax1.set_xlim(0, max_trans)
            ax1.legend(title="Run")
            sns.despine(ax=ax1)

            ax2.set_title("Loop Closure Success Rate vs Rotation Threshold")
            ax2.set_xlabel("Rotation Threshold (degrees)")
            ax2.set_ylabel("Success Rate (%)")
            ax2.set_ylim(0, 105)
            ax2.set_xlim(0, max_rot)
            ax2.legend(title="Run")
            sns.despine(ax=ax2)

        ax3.set_title("Loop Closures Under Translation Threshold")
        ax3.set_xlabel("Translation Threshold (m)")
        ax3.set_ylabel("Number of Loop Closures")
        ax3.set_xlim(0, max_trans)
        ax3.legend(title="Run")
        sns.despine(ax=ax3)

        ax4.set_title("Loop Closures Under Rotation Threshold")
        ax4.set_xlabel("Rotation Threshold (degrees)")
        ax4.set_ylabel("Number of Loop Closures")
        ax4.set_xlim(0, max_rot)
        ax4.legend(title="Run")
        sns.despine(ax=ax4)

        if include_rate_plots:
            ax5.set_title("Loop Closure Success Rate vs Combined Thresholds (1 m = 5°)")
            ax5.set_xlabel("Translation Threshold (m)")
            ax5.set_ylabel("Success Rate (%)")
            ax5.set_ylim(0, 105)
            ax5.set_xlim(0, max_trans)
            ax5_top = ax5.twiny()
            ax5_top.set_xlim(0, 5.0 * max_trans)
            ax5_top.set_xlabel("Rotation Threshold (degrees)")
            ax5.legend(title="Run")
            sns.despine(ax=ax5, trim=True)

        ax6.set_title("Loop Closures Under Combined Thresholds (1 m = 5°)")
        ax6.set_xlabel("Translation Threshold (m)")
        ax6.set_ylabel("Number of Loop Closures")
        ax6.set_xlim(0, max_trans)
        ax6_top = ax6.twiny()
        ax6_top.set_xlim(0, 5.0 * max_trans)
        ax6_top.set_xlabel("Rotation Threshold (degrees)")
        ax6.legend(title="Run")
        sns.despine(ax=ax6, trim=True)

        fig.tight_layout()

        if show_plots:
            plt.show()

        return fig
    
    @staticmethod
    def visualize_error_scatter(
        lc_data_list: List[LoopClosureData],
        labels: List[str],
        inlier_masks: List[np.ndarray] = None,
        group_indices: List[int] = None,
        show_plots: bool = True,
        save_path: str = None,
        max_translation_frac: float = 1.0,
        max_rotation_frac: float = 1.0,
        title: str = None,
        color_by_values: List[np.ndarray] = None,
        color_by_label: str = None,
        ax: plt.Axes = None,
        marker_size_x: float = 75,
        marker_size_star: float = 300,
    ):
        """
        Scatter plot of loop closure errors (log-log scale): each point is one
        loop closure. Inliers and outliers are shown with different markers.

        Args:
            lc_data_list: LoopClosureData instances, each with ``calculate_errors``
                and ``label_successful`` already called (i.e. ``.results`` populated
                with ``successful`` labels), all using the same
                ``trans_err_in_target``/``rot_err_in_target``. The highlighted
                target-box rectangle uses those shared thresholds.
            labels: Display name for each entry in ``lc_data_list``.
            inlier_masks: Optional list of boolean arrays marking inlier loop
                closures within each entry. Mutually exclusive with ``group_indices``.
            group_indices: Optional list of integers (one per entry in ``lc_data_list``)
                grouping entries into pairs. Within each group, the first occurrence
                is plotted as X markers (all loop closures) and the second as star
                markers (inliers). Paired entries share a color. Mutually exclusive
                with ``inlier_masks``.
            show_plots: If True, display the plot interactively.
            save_path: If provided, save the figure to this path instead of showing.
            max_translation_frac: Fraction of max translation error for axis limit.
            max_rotation_frac: Fraction of max rotation error for axis limit.
            title: Optional plot title.
            color_by_values: Optional list of per-loop-closure numeric arrays (one per
                entry in ``lc_data_list``). When provided, points are colored by their
                value using a sequential colormap instead of by label.
            color_by_label: Label for the colorbar shown when ``color_by_values`` is set.
            ax: Optional existing ``Axes`` to draw into. When provided the scatter is
                rendered directly into that axes (which must belong to an already-created
                figure); ``show_plots``, ``save_path``, ``tight_layout``, and
                ``plt.close`` are all skipped — the caller is responsible for saving and
                closing the figure. When ``None`` (default), a new 10×10 inch figure is
                created and the usual show/save/close logic applies unchanged.
            marker_size_x: Marker size (``s`` in ``scatter``) for the "all loop closures"
                X markers. The legend marker scales proportionally.
            marker_size_star: Marker size (``s`` in ``scatter``) for the inlier star
                markers. The legend marker scales proportionally.

        Returns:
            Tuple of (matplotlib Figure, list of stats dicts). Each stats dict contains
            ``"label"``, ``"success_rate"``, ``"num_successful_loop_closures"``, and
            ``"num_loop_closures"`` for one entry in ``lc_data_list``. When ``ax`` is
            provided the returned Figure is the caller-owned figure that ``ax`` belongs to.

        Raises:
            ValueError: If list lengths do not match, fraction values are out of range,
                both ``inlier_masks`` and ``group_indices`` are provided, ``ax`` is
                provided alongside ``show_plots`` or ``save_path``, or the entries in
                ``lc_data_list`` don't share the same ``trans_err_in_target``/``rot_err_in_target``.
            RuntimeError: If both ``show_plots`` and ``save_path`` are set.
        """

        if inlier_masks is not None and group_indices is not None:
            raise ValueError("inlier_masks and group_indices are mutually exclusive — pick one")

        if ax is not None and (show_plots or save_path is not None):
            raise ValueError("ax is mutually exclusive with show_plots and save_path — the caller owns the figure")

        if len(labels) != len(lc_data_list):
            raise ValueError("labels must have the same length as lc_data_list")

        if inlier_masks is not None and len(inlier_masks) != len(lc_data_list):
            raise ValueError("inlier_masks must have the same length as lc_data_list")

        if group_indices is not None and len(group_indices) != len(lc_data_list):
            raise ValueError("group_indices must have the same length as lc_data_list")

        if group_indices is not None:
            counts = Counter(group_indices)
            over = [g for g, c in counts.items() if c > 2]
            if over:
                raise ValueError(f"Each group index may appear at most twice; found >2 occurrences for: {over}")

        if color_by_values is not None and len(color_by_values) != len(lc_data_list):
            raise ValueError("color_by_values must have the same length as lc_data_list")

        if not (0 < max_translation_frac <= 1.0):
            raise ValueError("max_translation_frac must be in (0, 1]")

        if not (0 < max_rotation_frac <= 1.0):
            raise ValueError("max_rotation_frac must be in (0, 1]")

        trans_err_in_target: float = lc_data_list[0].results.trans_err_in_target
        rot_err_in_target: float = lc_data_list[0].results.rot_err_in_target
        mismatched: List[int] = []
        for i, lc in enumerate(lc_data_list):
            if lc.results.trans_err_in_target != trans_err_in_target or lc.results.rot_err_in_target != rot_err_in_target:
                mismatched.append(i)
        if mismatched:
            raise ValueError(
                f"All entries in lc_data_list must share the same trans_err_in_target/rot_err_in_target "
                f"(from label_successful); entries at indices {mismatched} don't match entry 0's "
                f"({trans_err_in_target}, {rot_err_in_target})."
            )

        using_group_indices = group_indices is not None
        if group_indices is not None:
            # Convert group_indices to inlier_masks: first occurrence of each group
            # gets all-zeros (all loop closures, X marker), second gets all-ones
            # (inliers, star marker). Entries within the same group share a color.
            seen_groups: dict = {}
            for gi in group_indices:
                if gi not in seen_groups:
                    seen_groups[gi] = len(seen_groups)
            group_occurrence: dict = {}
            is_inlier_entry = []
            inlier_masks = []
            for lc, gi in zip(lc_data_list, group_indices):
                occ = group_occurrence.get(gi, 0)
                group_occurrence[gi] = occ + 1
                n = len(np.asarray(lc.results.translation_errors))
                is_inlier_entry.append(occ >= 1)
                inlier_masks.append(np.ones(n, dtype=bool) if occ >= 1 else np.zeros(n, dtype=bool))
        else:
            is_inlier_entry = [False] * len(lc_data_list)
            if inlier_masks is None:
                inlier_masks = [None] * len(lc_data_list)

        sns.set_theme(style="whitegrid", context="talk", palette="tab10")
        sns.set_context("poster", font_scale=1.0)
        _owns_figure = ax is None
        if _owns_figure:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        if using_group_indices:
            group_palette = sns.color_palette("bright", len(seen_groups))
            palette = [group_palette[seen_groups[gi]] for gi in group_indices]
        else:
            palette = sns.color_palette("bright", len(lc_data_list))

        # Collect all translation and rotation errors
        all_trans = np.concatenate([np.asarray(lc.results.translation_errors) for lc in lc_data_list])
        all_rot = np.concatenate([np.asarray(lc.results.rotation_errors) for lc in lc_data_list])

        # Apply max fraction
        if all_trans.size > 0:
            all_trans = all_trans[all_trans <= np.max(all_trans) * max_translation_frac]
        if all_rot.size > 0:
            all_rot = all_rot[all_rot <= np.max(all_rot) * max_rotation_frac]

        # Determine axis limits with a small margin (e.g., 5%). Entries with no loop
        # closures at all (e.g. zero inliers) leave these arrays empty; fall back to a
        # unit range so the (empty) plot still renders instead of crashing on np.min/max.
        # Axes are log-scaled, so mins must stay strictly positive -- clamp to a small
        # epsilon rather than 0.0 (which would otherwise be silently ignored by
        # set_xlim/set_ylim below, since the actual min error can also be exactly 0).
        _log_axis_eps = 1e-6
        x_min = max(np.min(all_trans) * 0.95, _log_axis_eps) if all_trans.size > 0 else _log_axis_eps
        x_max = np.max(all_trans) * 1.05 if all_trans.size > 0 else 1.0
        y_min = max(np.min(all_rot) * 0.95, _log_axis_eps) if all_rot.size > 0 else _log_axis_eps
        y_max = np.max(all_rot) * 1.05 if all_rot.size > 0 else 1.0

        # When coloring by values, build a shared normalizer across all entries
        if color_by_values is not None:
            all_cbv = np.concatenate([np.asarray(v, dtype=float) for v in color_by_values])
            cbv_norm = Normalize(vmin=np.nanmin(all_cbv), vmax=np.nanmax(all_cbv))
            cbv_cmap = cm.get_cmap("viridis")
            sm = cm.ScalarMappable(norm=cbv_norm, cmap=cbv_cmap)
            sm.set_array([])

        legend_handles = []
        stats_list = []
        for idx, (lc, label, color, inlier_mask) in enumerate(zip(lc_data_list, labels, palette, inlier_masks)):
            trans_err = np.asarray(lc.results.translation_errors)
            rot_err = np.asarray(lc.results.rotation_errors)

            # 1. Calculate how many points fall inside the highlighted square
            num_in_box = np.sum(lc.results.successful)
            total_points = len(trans_err)

            # Avoid division by zero if a list is empty
            percent_in_box = (num_in_box / total_points * 100) if total_points > 0 else 0

            stats_list.append({
                "label": label,
                "success_rate": percent_in_box,
                "num_successful_loop_closures": int(num_in_box),
                "num_loop_closures": total_points,
            })

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
            else:
                num_inliers = 0

            # Determine per-point colors
            if color_by_values is not None:
                cbv = np.asarray(color_by_values[idx], dtype=float)
                point_colors_out = cbv_cmap(cbv_norm(cbv[outlier_mask]))
                point_colors_in = cbv_cmap(cbv_norm(cbv[inlier_vis_mask]))
                scatter_color_out = point_colors_out
                scatter_color_in = point_colors_in
            else:
                scatter_color_out = color
                scatter_color_in = color

            # Outliers
            ax.scatter(
                trans_err[outlier_mask], rot_err[outlier_mask],
                alpha=0.8, s=marker_size_x, color=scatter_color_out,
                marker='x', clip_on=False, zorder=5,
            )
            # Inliers (different marker)
            if np.any(inlier_vis_mask):
                ax.scatter(
                    trans_err[inlier_vis_mask], rot_err[inlier_vis_mask],
                    alpha=0.8, s=marker_size_star, color=scatter_color_in,
                    marker='*', edgecolors='none', clip_on=False, zorder=5,
                )

            tag = "[inliers] " if is_inlier_entry[idx] else ""
            #print(f"{tag}{label} ({percent_in_box:.1f}% ({num_in_box}/{total_points}) in target)")
            #if not using_group_indices:
                #print(f"Number of inliers for {label}: {num_inliers}")
            if color_by_values is not None and np.any(lc.results.successful):
                cbv_in_box = np.asarray(color_by_values[idx], dtype=float)[lc.results.successful]
                #print(f"Min {color_by_label or 'color_by'} in target box for {label}: {np.nanmin(cbv_in_box)}")
            if color_by_values is None and not is_inlier_entry[idx]:
                legend_handles.append(Patch(facecolor=color, edgecolor='none', label=label))

        if using_group_indices:
            legend_handles.append(Line2D([0], [0], marker='x', color='black', label='All loop closures',
                                         linestyle='None', markersize=10, markeredgewidth=2))
            legend_handles.append(Line2D([0], [0], marker='*', color='black', label='Inliers',
                                         linestyle='None', markersize=14))

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

                # --- LABEL VISIBILITY ---
                # Show labels for powers of 10 (1) and even numbers (2, 4, 6, 8)
                # Hide labels for odd numbers (3, 5, 7, 9)
                if leading_digit in [3, 5, 7, 9]:
                    tick.label1.set_visible(False)

        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if color_by_values is not None:
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(color_by_label if color_by_label is not None else "Value")
        else:
            ax.legend(title="Run", handles=legend_handles, frameon=True)
        sns.despine(ax=ax)
        if _owns_figure:
            fig.tight_layout()
            if show_plots and save_path is not None:
                raise RuntimeError("Can't enable both show_plots and save_path!")
            elif show_plots:
                plt.show()
            elif save_path is not None:
                plt.savefig(save_path)
            plt.close()

        return fig, stats_list
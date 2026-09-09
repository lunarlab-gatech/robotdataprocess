from .Data import CoordinateFrame, Data
from .LoopClosureData.LoopClosureData import LoopClosureData
from .LoopClosureData.LoopClosureDataResult import LoopClosureFilterMode
from .OdometryData import OdometryData
import copy
import itertools
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
from ..utils.ModuleImporter import ModuleImporter


class SLAMData(Data):
    """
    Holds one SLAM run's loaded data for one robot group -- trajectories, loop closures, and
    runtime/data-size/matcher stats. Built via :meth:`from_MeronomyGraph`.

    Attributes:
        system_params: The loaded SystemParams for this run.
        robot_names: Robot names in this group, sorted.
        pre_opt_est_trajectories: Pre-optimize estimated trajectories, in robot_names order.
        estimated_trajectories: Post-optimize estimated trajectories, in robot_names order.
        alignment_loop_closures: All (``LoopClosureFilterMode.ALL``) raw loop closures.
        inlier_loop_closures: All (``LoopClosureFilterMode.ALL``) per-pair inlier loop closures.
        timing: Runtime breakdown by category, then by the robot identity that produced it
            (see :meth:`load_timing_data`).
        data_size_mb: Estimated communication data size in decimal MB.
        mg_match: MG two-stage matcher stage-count/field stats, or ``None`` for non-MG runs.
    """

    system_params: Any
    robot_names: List[str]
    pre_opt_est_trajectories: List[OdometryData]
    estimated_trajectories: List[OdometryData]
    alignment_loop_closures: LoopClosureData
    inlier_loop_closures: LoopClosureData
    timing: Dict[str, Dict[Any, float]]
    data_size_mb: float
    mg_match: Optional[Dict]

    def __init__(self, system_params: Any, robot_names: List[str],
                estimated_trajectories: List[OdometryData],
                alignment_loop_closures: LoopClosureData, inlier_loop_closures: LoopClosureData,
                timing: Dict[str, Dict[Any, float]], data_size_mb: float,
                pre_opt_est_trajectories: Optional[List[OdometryData]] = None,
                mg_match: Optional[Dict] = None):
        super().__init__(frame_id='')
        self.system_params = system_params
        self.robot_names = list(robot_names)
        self.estimated_trajectories = estimated_trajectories
        self.alignment_loop_closures = alignment_loop_closures
        self.inlier_loop_closures = inlier_loop_closures
        self.timing = timing
        self.data_size_mb = data_size_mb
        self.pre_opt_est_trajectories = pre_opt_est_trajectories if pre_opt_est_trajectories is not None else []
        self.mg_match = mg_match

    @staticmethod
    def _assert_sorted_robot_names(robot_names: List[str]) -> None:
        """Every loader requires robot_names pre-sorted; only from_MeronomyGraph sorts for you."""
        if list(robot_names) != sorted(robot_names):
            raise ValueError(f"robot_names must be sorted, got {list(robot_names)!r}")

    # =========================================================================
    # ======================= Helper Loader Methods ===========================
    # =========================================================================
    @staticmethod
    def ensure_MeronomyGraph_importable(mg_root: Path) -> None:
        """
        Adds mg_root to sys.path if not already present, so MeronomyGraph is importable.

        Must be called in the parent process before creating any multiprocessing Pool whose
        workers will return MeronomyGraph-backed objects (e.g. the SystemParams cached on
        SLAMData) -- otherwise unpickling those objects back in the parent raises
        ModuleNotFoundError, since fork-based workers each add mg_root to their own sys.path
        independently of the parent's.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
        """
        if str(mg_root) not in sys.path:
            sys.path.insert(0, str(mg_root))

    @staticmethod
    def load_system_params(mg_root: Path, dataset_name: str, dataset_seq: str, method: str) -> Any:
        """
        Loads the SystemParams for one experiment config, used to reconstruct hash-addressed result
        directories. dataset_name/dataset_seq are stored on the returned SystemParams.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            dataset_name: The dataset used.
            dataset_seq: The dataset version/sequence.
            method: Run name identifying which experiment config to load.

        Returns:
            The loaded SystemParams.
        """
        SLAMData.ensure_MeronomyGraph_importable(mg_root)
        SystemParams = ModuleImporter.get_module_attribute('MeronomyGraph.params.system_params', 'SystemParams')

        return SystemParams.from_experiment_config(str(mg_root / "params"), dataset_name, dataset_seq, method)

    @staticmethod
    def load_est_data(mg_root: Path, system_params: Any,
                robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> List[OdometryData]:
        """
        Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            List of OdometryData in the same order as robot_names.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        rpgo_dir = system_params.rpgo_result_dir(mg_root / "results",
                                                robot_names, critical_invocation_params)
        return [
            OdometryData.from_csv(
                str(rpgo_dir / f'{rn}.csv'),
                "map", 'robot' + str(i), CoordinateFrame.NONE, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
            for i, rn in enumerate(robot_names)
        ]

    @staticmethod
    def load_kimera_rpgo_first_stage_est_data(mg_root: Path, system_params: Any,
                                                    robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> List[OdometryData]:
        """
        Load pre-optimize (first-stage) estimated trajectories for a set of robots.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            List of OdometryData in the same order as robot_names.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        rpgo_dir = system_params.rpgo_result_dir(mg_root / "results",
                                                robot_names, critical_invocation_params)
        names_override = {chr(97 + i): name for i, name in enumerate(robot_names)}
        return [
            OdometryData.from_g2o(str(rpgo_dir / 'pre_optimize' / 'result.g2o'), str(rpgo_dir / 'dense' / 'odom_all.time.txt'), rn,
                "map", 'robot' + str(i), CoordinateFrame.NONE, names_override)
            for i, rn in enumerate(robot_names)
        ]

    @staticmethod
    def load_LC_data(mg_root: Path, system_params: Any, robot_names: List[str],
                        critical_invocation_params: Dict[str, Any],
                        names_override: Optional[Dict] = None) -> Tuple[LoopClosureData, LoopClosureData]:
        """
        Load all (``LoopClosureFilterMode.ALL``) LC data for a ROMAN run. Use
        :meth:`get_loop_closures` to derive an inter-/intra-only subset without re-reading disk.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.
            names_override: Maps g2o character keys ('a', 'b', ...) to the robot names used in the
                returned data (default: robot_names) -- pass display names when LC names must
                match a visualize_2D nameList.

        Returns:
            (merged_lc, merged_lc_inlier), both in ``LoopClosureFilterMode.ALL``.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        rpgo_dir = system_params.rpgo_result_dir(mg_root / "results",
                                                robot_names, critical_invocation_params)
        letter_by_name = {name: chr(97 + i) for i, name in enumerate(robot_names)}

        effective_override = names_override if names_override is not None else \
            {chr(97 + i): name for i, name in enumerate(robot_names)}

        # odom_and_lc.g2o already contains all robot pairs — load it once to avoid
        # tripling the count when iterating over combinations_with_replacement.
        merged_lc = LoopClosureData.from_g2o(
            rpgo_dir / 'dense' / 'odom_and_lc.g2o',
            rpgo_dir / 'dense' / 'odom_all.time.txt',
            names_override=effective_override)

        # Load the per-pair inlier g2o files (these are pair-specific)
        # Kimera-RPGO writes these against the sparse-keyframe-indexed graph when sparsified, so they need sparse/odom_all.time.txt, not dense.
        inlier_time_subdir = 'sparse' if system_params.offline_rpgo_params.sparsified else 'dense'
        lc_inlier_data_list = []
        for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):
            letter_a = letter_by_name[name_a]
            letter_b = letter_by_name[name_b]
            if name_a == name_b:
                g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
            else:
                g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
            lc_data_inlier = LoopClosureData.from_g2o(rpgo_dir / g2o_filename,
                                                    rpgo_dir / inlier_time_subdir / 'odom_all.time.txt',
                                                    names_override=effective_override)
            lc_inlier_data_list.append(lc_data_inlier)

        merged_lc_inlier = LoopClosureData.merge(lc_inlier_data_list)

        merged_lc.prune_duplicates()
        merged_lc_inlier.prune_duplicates()

        return merged_lc, merged_lc_inlier

    @staticmethod
    def load_timing_data(mg_root: Path, system_params: Any,
                            robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Load the runtime (s) breakdown for a ROMAN run on a robot group, keyed by the robot
        identity that produced it (not by file path) so :meth:`total_data_generation_time` can
        dedup work shared between groups without further disk access.

        Reads each combination's own alignment runtime (``<robot_a>_<robot_b>.runtime.txt``, one line,
        at that combination's own align result dir), each robot's own mapping runtime
        (``<robot>.runtime.txt``, at its own mapping result dir), and the offline RPGO runtime
        (``runtime.txt``, a single value on its last non-empty line, at the rpgo result dir).

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            Dict with keys ``"align"`` (``{(robot_a, robot_b): seconds}``, one entry per
            combination including self-pairs), ``"mapping"`` (``{robot: seconds}``), and
            ``"offline_rpgo"`` (``{tuple(robot_names): seconds}``).

        Raises:
            FileNotFoundError: If any runtime file is missing.
            ValueError: If any runtime file is empty.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        results_root = mg_root / "results"

        align_paths = {}
        for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):
            align_dir = system_params.align_result_dir(results_root,
                                                        name_a, name_b, critical_invocation_params)
            align_paths[(name_a, name_b)] = align_dir / f'{name_a}_{name_b}.runtime.txt'

        mapping_paths = {
            rn: system_params.mapping_result_dir(results_root, rn, critical_invocation_params) / f'{rn}.runtime.txt'
            for rn in robot_names
        }

        rpgo_runtime_path = system_params.rpgo_result_dir(results_root,
                                                        robot_names, critical_invocation_params) / 'runtime.txt'

        missing = [p for p in [*align_paths.values(), *mapping_paths.values(), rpgo_runtime_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing runtime file: {missing[0]}")

        align_lines = {pair: p.read_text().strip() for pair, p in align_paths.items()}
        mapping_lines = {rn: p.read_text().strip() for rn, p in mapping_paths.items()}
        rpgo_lines = [line.strip() for line in rpgo_runtime_path.read_text().splitlines() if line.strip()]
        if not all(align_lines.values()) or not all(mapping_lines.values()) or not rpgo_lines:
            raise ValueError(f"Empty runtime file for robot group {robot_names}")

        align = {pair: float(line.split(':')[-1]) for pair, line in align_lines.items()}
        mapping = {rn: float(line.split(':')[-1]) for rn, line in mapping_lines.items()}
        offline_rpgo = {tuple(robot_names): float(rpgo_lines[-1])}

        return {"align": align, "mapping": mapping, "offline_rpgo": offline_rpgo}

    @staticmethod
    def load_data_size(mg_root: Path, system_params: Any,
                            robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> float:
        """
        Load the total estimated communication data size (decimal MB, 1 MB = 1,000,000 bytes)
        for a ROMAN run across a group of robots.

        Sums ``align.data_size.txt`` (a single ``"Total submap data size (bytes): <value>"``
        line) across every inter-robot combination within the group -- unlike
        :meth:`load_timing_data`'s pairing, self-pairs are excluded, since a robot doesn't send
        itself any data.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            The total data size in decimal MB (not MiB).

        Raises:
            FileNotFoundError: If any combination's data size file is missing.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        results_root = mg_root / "results"

        data_size_paths = []
        for name_a, name_b in itertools.combinations(robot_names, 2):
            align_dir = system_params.align_result_dir(results_root,
                                                        name_a, name_b, critical_invocation_params)
            data_size_paths.append(align_dir / 'align.data_size.txt')

        missing = [p for p in data_size_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing data size file: {missing[0]}")

        total_bytes = sum(float(p.read_text().strip().split(':')[-1]) for p in data_size_paths)
        return total_bytes / 1_000_000

    @staticmethod
    def load_mg_match_stats(mg_root: Path, system_params: Any,
                                robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> Optional[Dict]:
        """
        Count MG two-stage matcher calls by stage for a robot group.

        Reads ``align.mg_match.txt`` for each intra-/inter-robot combination, at each
        combination's own align result dir, and tallies how many calls reached stage 0, 1, or 2.
        Also collects, across all stage-1/2 calls, the values of ``n_stage1_matches`` and, across
        all stage-2 calls, ``n_stage2_child_clipper``,
        ``n_stage2_unmatched_children_to_parents_clipper``,
        ``n_stage2_unmatched_children_to_children_clipper``, and ``stage2_point_error``. Fields
        absent from a given line (older log formats don't include all fields) are skipped for that
        line rather than raising. Files that don't exist (e.g. non-MG methods) are skipped.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            Dict with ``"stage_counts"`` (stage -> occurrence count) and one list of values per
            collected field name above (``stage2_point_error`` as floats, possibly ``nan``; the
            rest as ints), or ``None`` if no ``align.mg_match.txt`` files exist (e.g. a non-MG run).
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        results_root = mg_root / "results"

        stage1_fields = ["n_stage1_matches"]
        stage2_fields = [
            "n_stage2_child_clipper",
            "n_stage2_unmatched_children_to_parents_clipper",
            "n_stage2_unmatched_children_to_children_clipper",
        ]
        field_types = {field: int for field in stage1_fields + stage2_fields}
        field_types["stage2_point_error"] = float
        stage2_fields = stage2_fields + ["stage2_point_error"]

        stage_counts = {0: 0, 1: 0, 2: 0}
        field_values = {field: [] for field in stage1_fields + stage2_fields}
        found_any = False
        for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):
            align_dir = system_params.align_result_dir(results_root,
                                                        name_a, name_b, critical_invocation_params)
            mg_match_path = align_dir / 'align.mg_match.txt'
            if not mg_match_path.exists():
                continue
            found_any = True
            for line in mg_match_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                stage = int(tokens[1])
                stage_counts[stage] += 1
                fields = stage1_fields if stage == 1 else stage1_fields + stage2_fields if stage == 2 else []
                for field in fields:
                    key = field + ':'
                    if key in tokens:
                        field_values[field].append(field_types[field](tokens[tokens.index(key) + 1]))

        return {"stage_counts": stage_counts, **field_values} if found_any else None

    # =========================================================================
    # ============================ Getter Methods =============================
    # =========================================================================
    def get_loop_closures(self, lc_filter: LoopClosureFilterMode) -> Tuple[LoopClosureData, LoopClosureData]:
        """
        Returns ``(alignment_loop_closures, inlier_loop_closures)`` restricted to ``lc_filter``,
        derived from the already-loaded ``LoopClosureFilterMode.ALL`` data -- no disk I/O.

        Args:
            lc_filter: Which subset of loop closures to return.

        Returns:
            A pair of new LoopClosureData instances; the cached ``ALL``-mode data is never mutated.
        """
        alignment_lc = copy.deepcopy(self.alignment_loop_closures)
        inlier_lc = copy.deepcopy(self.inlier_loop_closures)
        if lc_filter == LoopClosureFilterMode.ONLY_INTER_LC:
            alignment_lc.prune_intra_robot_loop_closures()
            inlier_lc.prune_intra_robot_loop_closures()
        elif lc_filter == LoopClosureFilterMode.ONLY_INTRA_LC:
            alignment_lc.prune_inter_robot_loop_closures()
            inlier_lc.prune_inter_robot_loop_closures()
        return alignment_lc, inlier_lc

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================
    @classmethod
    def from_MeronomyGraph(cls, mg_root: Path, dataset_name: str, dataset_seq: str, method: str,
                           robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> 'SLAMData':
        """
        Loads a SLAMData instance from a MeronomyGraph result tree, for one run (method) on one
        robot group. This is the only method on this class that accepts unsorted robot_names.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            dataset_name: Which dataset this run is for (e.g. "hercules", "airmuseum").
            dataset_seq: The dataset version/sequence (e.g. "V2.4.F", "Scenario5").
            method: Run name identifying which experiment config to load.
            robot_names: Robot names in this group, in any order.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            A populated SLAMData instance. ``pre_opt_est_trajectories`` is ``[]`` if the
            pre-optimize files are unavailable, and ``mg_match`` is ``None`` for non-MG runs.
        """
        sorted_robot_names = sorted(robot_names)

        system_params = cls.load_system_params(mg_root, dataset_name, dataset_seq, method)
        estimated_trajectories = cls.load_est_data(mg_root, system_params, sorted_robot_names,
                                                   critical_invocation_params)

        try:
            pre_opt_est_trajectories = cls.load_kimera_rpgo_first_stage_est_data(
                mg_root, system_params, sorted_robot_names, critical_invocation_params)
        except Exception as e:
            print(f"Warning: Could not load first-stage trajectories for {dataset_seq} {method}: {e}")
            pre_opt_est_trajectories = []

        alignment_loop_closures, inlier_loop_closures = cls.load_LC_data(
            mg_root, system_params, sorted_robot_names, critical_invocation_params)
        timing = cls.load_timing_data(mg_root, system_params, sorted_robot_names, critical_invocation_params)
        data_size_mb = cls.load_data_size(mg_root, system_params, sorted_robot_names, critical_invocation_params)
        mg_match = cls.load_mg_match_stats(mg_root, system_params, sorted_robot_names, critical_invocation_params)

        return cls(system_params, sorted_robot_names, estimated_trajectories,
                   alignment_loop_closures, inlier_loop_closures, timing, data_size_mb,
                   pre_opt_est_trajectories=pre_opt_est_trajectories, mg_match=mg_match)

    # =========================================================================
    # ======================== Multi SLAMData Methods =========================
    # =========================================================================
    @staticmethod
    def get_timing_totals(slam_data_list: List['SLAMData']) -> Dict[str, float]:
        """
        Per-category runtime (s) totals across the given instances, deduplicated by the robot
        identity keys :meth:`load_timing_data` set -- the SLAM pipeline caches, so a robot's
        mapping and a pair's alignment shared by several groups only ran once. Deduplication is a
        no-op for a single instance, where each entry already appears exactly once.

        Args:
            slam_data_list: SLAMData instances to total, e.g. every robot group for one run. Pass
                one instance for that group's own totals; sum the returned values for a run's
                total data generation time.

        Returns:
            Dict with keys ``"align"``, ``"mapping"``, and ``"offline_rpgo"``, in seconds.

        Raises:
            ValueError: If two instances report different runtimes for the same robot identity,
                which would mean they came from different underlying results.
        """
        merged: Dict[str, Dict[Any, float]] = {"align": {}, "mapping": {}, "offline_rpgo": {}}
        for slam_data in slam_data_list:
            for category in merged:
                for key, seconds in slam_data.timing[category].items():
                    if key in merged[category] and merged[category][key] != seconds:
                        raise ValueError(f"Conflicting {category} runtime for {key}: "
                                         f"{merged[category][key]} != {seconds}")
                    merged[category][key] = seconds

        return {category: sum(entries.values()) for category, entries in merged.items()}
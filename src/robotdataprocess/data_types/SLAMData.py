from .Data import CoordinateFrame, Data
from .LoopClosureData.LoopClosureData import LoopClosureData
from .LoopClosureData.LoopClosureDataResult import LoopClosureFilterMode
from .OdometryData import OdometryData
import copy
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
from ..utils.graph_utils import kruskal_connection_weights
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
        data_size: Each inter-robot pair's cumulative communication data size (bytes) and objects sent (or
            ``None`` if not recorded) at the end of each alignment attempt (see :meth:`load_data_size`).
        mg_match: MG two-stage matcher stage-count/field stats, or ``None`` for non-MG runs.
        connection_attempts: For SlideSLAM runs, the attempt at which each (robot_a, robot_b) pair in this group
            first became connected by the loop closures it uses (transitive chaining); ``None`` for other methods,
            which search every pair to the end.
    """

    system_params: Any
    robot_names: List[str]
    pre_opt_est_trajectories: List[OdometryData]
    estimated_trajectories: List[OdometryData]
    alignment_loop_closures: LoopClosureData
    inlier_loop_closures: LoopClosureData
    timing: Dict[str, Dict[Any, Any]]
    data_size: Dict[Tuple, Tuple[List[float], Optional[List[int]]]]
    mg_match: Optional[Dict]
    connection_attempts: Optional[Dict[Tuple[str, str], int]]

    def __init__(self, system_params: Any, robot_names: List[str],
                estimated_trajectories: List[OdometryData],
                alignment_loop_closures: LoopClosureData, inlier_loop_closures: LoopClosureData,
                timing: Dict[str, Dict[Any, Any]], data_size: Dict[Tuple, Tuple[List[float], Optional[List[int]]]],
                pre_opt_est_trajectories: Optional[List[OdometryData]] = None,
                mg_match: Optional[Dict] = None, connection_attempts: Optional[Dict[Tuple[str, str], int]] = None):
        super().__init__(frame_id='')
        self.system_params = system_params
        self.robot_names = list(robot_names)
        self.estimated_trajectories = estimated_trajectories
        self.alignment_loop_closures = alignment_loop_closures
        self.inlier_loop_closures = inlier_loop_closures
        self.timing = timing
        self.data_size = data_size
        self.pre_opt_est_trajectories = pre_opt_est_trajectories if pre_opt_est_trajectories is not None else []
        self.mg_match = mg_match
        self.connection_attempts = connection_attempts

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
    def load_system_params(mg_root: Path, dataset_name: str, dataset_seq: str, method: str,
                           param_overrides: Optional[Dict[str, Any]] = None) -> Any:
        """
        Loads the SystemParams for one experiment config, used to reconstruct hash-addressed result
        directories. dataset_name/dataset_seq are stored on the returned SystemParams.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            dataset_name: The dataset used.
            dataset_seq: The dataset version/sequence.
            method: Run name identifying which experiment config to load.
            param_overrides: Dot-notation param overrides the run was generated with, applied before
                the result directory hashes are computed; None for none.

        Returns:
            The loaded SystemParams.
        """
        SLAMData.ensure_MeronomyGraph_importable(mg_root)
        SystemParams = ModuleImporter.get_module_attribute('MeronomyGraph.params.system_params', 'SystemParams')

        return SystemParams.from_experiment_config(str(mg_root / "params"), dataset_name, dataset_seq, method,
                                                   overrides=param_overrides)

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
    def load_slideslam_accepted_attempt(lc_json_path: str, alignment_method: Any) -> Optional[int]:
        """
        The attempt at which SlideSLAM accepted a pair's loop closure, from an align.json written by
        MeronomyGraph's save_submap_align_results, for transitive chaining. Restricted to SlideSLAM methods
        because they write at most one loop closure per pair; other methods can write many, which would need
        matching each entry to its edge in align.g2o rather than reading a single attempt.

        Args:
            lc_json_path (str): Path to the pair's align.json.
            alignment_method (AlignmentMethod): The run's alignment method; must be SLIDEMATCH or SLIDEGRAPH.
        Returns:
            Optional[int]: The accepted attempt, or None if the pair has no loop closure.
        """
        if not alignment_method.is_slideslam:
            raise ValueError(f"Accepted attempts are only meaningful for SlideSLAM methods, got {alignment_method.name}")
        with open(lc_json_path) as f:
            entries: List[dict] = json.load(f)
        return entries[0]['attempt'] if entries else None

    @staticmethod
    def _parse_align_runtime(text: str) -> List[float]:
        """
        Reads the per-attempt alignment runtimes from a ``<robot_a>_<robot_b>.runtime.txt`` file. The current
        format has one line per attempt, e.g. ``a_b attempt 2: 4.123456789 load 0.52 0.48 0.40`` (the load
        averages are ignored); older files have a single ``<label>: <seconds>`` line, read as one attempt.

        Args:
            text: The runtime file's contents.
        Returns:
            List[float]: The seconds of each attempt, in attempt order.
        """

        # Keep the non-blank lines
        lines: List[str] = []
        for line in text.splitlines():
            if line.strip():
                lines.append(line.strip())

        # Older format: a single "<label>: <seconds>" line
        if ' attempt ' not in lines[0]:
            return [float(lines[0].split(':')[-1])]

        # Current format: "<pair> attempt <n>: <seconds> load <1min> <5min> <15min>" per attempt
        attempt_seconds: List[float] = []
        for line in lines:
            after_colon: str = line.split(':', 1)[1]           # " <seconds> load <1min> <5min> <15min>"
            seconds_text: str = after_colon.split('load')[0]  # " <seconds> "
            attempt_seconds.append(float(seconds_text))
        return attempt_seconds

    @staticmethod
    def _slideslam_connection_attempts(mg_root: Path, system_params: Any, robot_names: List[str],
                                       critical_invocation_params: Dict[str, Any]) -> Dict[Tuple[str, str], int]:
        """
        For a SlideSLAM run, the attempt at which each pair of robots in this group first became connected
        through the loop closures SlideSLAM uses (transitive chaining); after it, SlideSLAM would stop searching
        that pair, including pairs whose own loop closure was never accepted.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run; its alignment method must be SlideSLAM's.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.
        Returns:
            Dict[Tuple[str, str], int]: Each connected (robot_a, robot_b) pair, robot_a < robot_b, mapped to
                its connecting attempt. Pairs never connected are absent.
        """
        alignment_method: Any = system_params.submap_align_params.alignment_method
        if not alignment_method.is_slideslam:
            raise ValueError(f"Connection attempts only apply to SlideSLAM methods, got {alignment_method.name}")

        # Only accepted pairs are edges: a pair SlideSLAM never accepted didn't link its robots
        accepted_attempts: Dict[Tuple[str, str], int] = {}
        for name_a, name_b in itertools.combinations(robot_names, 2):
            align_dir: Path = system_params.align_result_dir(mg_root / "results", name_a, name_b, critical_invocation_params)
            attempt: Optional[int] = SLAMData.load_slideslam_accepted_attempt(str(align_dir / 'align.json'), alignment_method)
            if attempt is not None:
                accepted_attempts[(name_a, name_b)] = attempt

        # Pairs are connected at the attempt of the earliest-found loop closures that join them
        return kruskal_connection_weights(accepted_attempts)

    @staticmethod
    def load_timing_data(mg_root: Path, system_params: Any,
                            robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Load the runtime (s) breakdown for a ROMAN run on a robot group, keyed by the robot
        identity that produced it (not by file path), qualified by dataset, so
        :meth:`get_timing_totals` can dedup work shared between groups of the same dataset
        without conflating same-named robots from different datasets/sequences.

        Reads each combination's own alignment runtime (``<robot_a>_<robot_b>.runtime.txt``, one line per
        attempt or an older single line, at that combination's own align result dir), each robot's own
        mapping runtime (``<robot>.runtime.txt``, at its own mapping result dir), and the offline RPGO
        runtime (``runtime.txt``, a single value on its last non-empty line, at the rpgo result dir).

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            Dict with keys ``"align"`` (``{(dataset_name, dataset_seq, robot_a, robot_b): [seconds per attempt]}``,
            the measured compute of every attempt run, one entry per combination including self-pairs),
            ``"mapping"`` (``{(dataset_name, dataset_seq, robot): seconds}``), and ``"offline_rpgo"``
            (``{(dataset_name, dataset_seq, *robot_names): seconds}``).

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

        dataset_key = (system_params.dataset_name, system_params.dataset_version)
        mapping = {(*dataset_key, rn): float(line.split(':')[-1]) for rn, line in mapping_lines.items()}
        offline_rpgo = {(*dataset_key, *robot_names): float(rpgo_lines[-1])}

        # Alignment: each pair's measured seconds per attempt (the same in every group sharing the pair)
        align: Dict[Tuple, List[float]] = {}
        for pair, text in align_lines.items():
            align[(*dataset_key, *pair)] = SLAMData._parse_align_runtime(text)

        return {"align": align, "mapping": mapping, "offline_rpgo": offline_rpgo}

    @staticmethod
    def load_data_size(mg_root: Path, system_params: Any,
                            robot_names: List[str], critical_invocation_params: Dict[str, Any]) -> Dict[Tuple, Tuple[List[float], Optional[List[int]]]]:
        """
        Load each inter-robot combination's estimated communication data size (bytes) and number of
        submap objects sent, as cumulative totals at the end of each alignment attempt, from its
        ``align.data_size.txt``. Self-pairs are excluded, unlike :meth:`load_timing_data`'s pairing,
        since a robot doesn't send itself any data. Files without attempt lines (methods that run once,
        and older files) are read as a single attempt equal to their totals.

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            system_params: The loaded SystemParams for this run.
            robot_names: Robot names in this group, already sorted.
            critical_invocation_params: Other data-affecting args from the original run invocation.

        Returns:
            ``{(dataset_name, dataset_seq, robot_a, robot_b): ([cumulative bytes per attempt],
            [cumulative objects sent per attempt] or None if the file doesn't record them)}``.

        Raises:
            FileNotFoundError: If any combination's data size file is missing.
            ValueError: If any combination's data size file has an unrecognized line label, is
                missing the data size line, or has attempt lines inconsistent with its totals.
        """
        SLAMData._assert_sorted_robot_names(robot_names)
        results_root = mg_root / "results"

        data_size_paths: Dict[Tuple[str, str], Path] = {}
        for name_a, name_b in itertools.combinations(robot_names, 2):
            align_dir = system_params.align_result_dir(results_root,
                                                        name_a, name_b, critical_invocation_params)
            data_size_paths[(name_a, name_b)] = align_dir / 'align.data_size.txt'

        missing = [p for p in data_size_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing data size file: {missing[0]}")

        dataset_key = (system_params.dataset_name, system_params.dataset_version)
        data_size: Dict[Tuple, Tuple[List[float], Optional[List[int]]]] = {}
        for pair, path in data_size_paths.items():

            # Each file has a "Total submap data size (bytes)" line, an optional "Total number of objects sent" line
            # (older files lack it), and for methods with several attempts, "Attempt <n> data size (bytes)" and
            # "Attempt <n> number of objects sent" lines in attempt order, with the cumulative totals after each attempt
            total_bytes: Optional[float] = None
            total_num_objects: Optional[int] = None
            attempt_bytes: List[float] = []
            attempt_num_objects: List[int] = []
            for line in path.read_text().strip().splitlines():
                label, value = line.split(':')
                label = label.strip()
                value = value.strip()
                if label == 'Total submap data size (bytes)':
                    total_bytes = float(value)
                elif label == 'Total number of objects sent':
                    total_num_objects = int(value)
                elif label.startswith('Attempt ') and label.endswith(' data size (bytes)'):
                    attempt_bytes.append(float(value))
                elif label.startswith('Attempt ') and label.endswith(' number of objects sent'):
                    attempt_num_objects.append(int(value))
                else:
                    raise ValueError(f"Unrecognized line label {label!r} in {path}")
            if total_bytes is None:
                raise ValueError(f"Data size file {path} is missing the data size line")

            # No attempt lines (methods that run once, and older files): a single attempt equal to the totals,
            # with objects sent unknown if the file predates that line
            if not attempt_bytes:
                num_objects: Optional[List[int]] = None if total_num_objects is None else [total_num_objects]
                data_size[(*dataset_key, *pair)] = ([total_bytes], num_objects)

            # Attempt lines: they come in bytes/objects pairs, and the totals must be the last attempt's cumulative values
            else:
                if len(attempt_num_objects) != len(attempt_bytes):
                    raise ValueError(f"Data size file {path} has {len(attempt_bytes)} attempt data size lines "
                                     f"but {len(attempt_num_objects)} attempt objects-sent lines")
                if attempt_bytes[-1] != total_bytes or attempt_num_objects[-1] != total_num_objects:
                    raise ValueError(f"Data size file {path}: totals ({total_bytes} bytes, {total_num_objects} objects) differ "
                                     f"from the last attempt's ({attempt_bytes[-1]} bytes, {attempt_num_objects[-1]} objects)")
                data_size[(*dataset_key, *pair)] = (attempt_bytes, attempt_num_objects)

        return data_size

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
                           robot_names: List[str], critical_invocation_params: Dict[str, Any],
                           param_overrides: Optional[Dict[str, Any]] = None) -> 'SLAMData':
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
            param_overrides: Dot-notation param overrides the run was generated with; None for none.

        Returns:
            A populated SLAMData instance. ``pre_opt_est_trajectories`` is ``[]`` if the
            pre-optimize files are unavailable, and ``mg_match`` is ``None`` for non-MG runs.
        """
        sorted_robot_names = sorted(robot_names)

        system_params = cls.load_system_params(mg_root, dataset_name, dataset_seq, method, param_overrides)
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
        data_size = cls.load_data_size(mg_root, system_params, sorted_robot_names, critical_invocation_params)
        mg_match = cls.load_mg_match_stats(mg_root, system_params, sorted_robot_names, critical_invocation_params)

        # SlideSLAM only: when this group's chaining connected each pair (other methods search every pair to the end)
        connection_attempts: Optional[Dict[Tuple[str, str], int]] = None
        if system_params.submap_align_params.alignment_method.is_slideslam:
            connection_attempts = cls._slideslam_connection_attempts(mg_root, system_params, sorted_robot_names,
                                                                     critical_invocation_params)

        return cls(system_params, sorted_robot_names, estimated_trajectories,
                   alignment_loop_closures, inlier_loop_closures, timing, data_size,
                   pre_opt_est_trajectories=pre_opt_est_trajectories, mg_match=mg_match,
                   connection_attempts=connection_attempts)

    def _num_attempts_used(self, name_a: str, name_b: str, num_attempts_run: int) -> int:
        """
        How many of a pair's attempts count in this group: all of them, except that for SlideSLAM a pair whose
        robots became connected (connection_attempts) stops at that attempt, since it stops searching connected pairs.
        """
        if self.connection_attempts is not None and (name_a, name_b) in self.connection_attempts:
            return min(self.connection_attempts[(name_a, name_b)], num_attempts_run)
        return num_attempts_run

    def align_time_used(self) -> float:
        """
        Alignment time (s) the method would have spent in this group: each pair's attempts that count here
        (see _num_attempts_used). Unlike timing["align"], this can differ between groups.
        """
        total: float = 0.0
        for (_, _, name_a, name_b), attempt_seconds in self.timing["align"].items():
            total += sum(attempt_seconds[:self._num_attempts_used(name_a, name_b, len(attempt_seconds))])
        return total

    def data_size_used(self) -> Tuple[float, Optional[int]]:
        """
        Communication data the method would have sent in this group: each inter-robot pair's cumulative data size
        and objects sent at its last attempt that counts here (see _num_attempts_used), summed over pairs.

        Returns:
            Tuple[float, Optional[int]]: (data size in decimal MB, 1 MB = 1,000,000 bytes; objects sent, or None
                if any pair's data size file doesn't record them).
        """
        total_bytes: float = 0.0
        total_num_objects: Optional[int] = 0
        for (_, _, name_a, name_b), (attempt_bytes, attempt_num_objects) in self.data_size.items():

            # Cumulative values, so take the one at the last counted attempt
            last_used: int = self._num_attempts_used(name_a, name_b, len(attempt_bytes)) - 1
            total_bytes += attempt_bytes[last_used]
            if attempt_num_objects is None or total_num_objects is None:
                total_num_objects = None
            else:
                total_num_objects += attempt_num_objects[last_used]
        return total_bytes / 1_000_000, total_num_objects

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
        merged: Dict[str, Dict[Any, Any]] = {"align": {}, "mapping": {}, "offline_rpgo": {}}
        for slam_data in slam_data_list:
            for category in merged:
                for key, seconds in slam_data.timing[category].items():
                    if key in merged[category] and merged[category][key] != seconds:
                        raise ValueError(f"Conflicting {category} runtime for {key}: "
                                         f"{merged[category][key]} != {seconds}")
                    merged[category][key] = seconds

        # Each alignment entry is a pair's seconds per attempt, so sum its attempts too
        align_total: float = 0.0
        for attempt_seconds in merged["align"].values():
            align_total += sum(attempt_seconds)

        # Mapping and offline RPGO hold one value per robot / group
        return {"align": align_total, "mapping": sum(merged["mapping"].values()),
                "offline_rpgo": sum(merged["offline_rpgo"].values())}
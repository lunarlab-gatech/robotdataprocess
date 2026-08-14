from dataclasses import dataclass, field
from enum import Enum
from evo.core.units import Unit
import itertools
import math
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import fitz
import pandas as pd
import re
from robotdataprocess import CoordinateFrame, LoopClosureData, OdometryData, PathData, PathDataAlignResult, TableData
import seaborn as sns
import sys
from typing import Any, Dict, List, Optional, Tuple

def load_system_params_ROMAN(roman_root: Path, dataset_prefix: str, dataset_name: str, method: str):
    """
    Loads the SystemParams for one experiment config, used to reconstruct hash-addressed result
    directories. Call once per (dataset_prefix, dataset_name, method) and pass into load_*_ROMAN.

    Args:
        roman_root: Path to the roman repo checkout.
        dataset_prefix: The dataset used.
        dataset_name: The dataset version/sequence.
        method: Run name identifying which experiment config to load.

    Returns:
        The loaded SystemParams.
    """
    roman_root = Path(roman_root)
    if str(roman_root) not in sys.path:
        sys.path.insert(0, str(roman_root))
    from roman.params.data_params import DataParams
    from roman.params.system_params import SystemParams

    experiment_path = roman_root / "params" / "experiments" / dataset_prefix / f"{dataset_name}_{method}.yaml"
    placeholder_data_params = DataParams(_img_data=None, _depth_data=None, _pose_data=None,
                                         img_data_params=None, T_camera_flu=np.eye(4))
    return SystemParams.from_experiment_config(str(experiment_path), {"_placeholder": placeholder_data_params})

def _make_raw_df(run_names: List[str], run_display_names: Dict[str, str],
                 cols_for_run, value_fn) -> pd.DataFrame:
    """Build a raw numeric DataFrame indexed by run display name.

    Args:
        run_names: Ordered list of run identifiers to use as rows.
        run_display_names: Maps each run identifier to its display name (row label).
        cols_for_run: callable(run) -> iterable of column labels for that run.
        value_fn: callable(run, col) -> float; return float('nan') for missing/
            suppressed cells.
    """
    return pd.DataFrame(
        {run_display_names.get(run, run): {col: value_fn(run, col) for col in cols_for_run(run)}
         for run in run_names}
    ).T

def make_highlighted_table(raw_df: pd.DataFrame, title: str, color_fn=None, fmt=None,
                   emphasis_rankings: bool = True, higher_is_better: bool = True) -> TableData:
    """Resolve a raw numeric DataFrame into a styled TableData, one column at a time.

    Missing/suppressed values must be represented as NaN in raw_df; fmt and
    color_fn are responsible for rendering NaN however the caller wants
    (e.g. "---"), since TableData._rank_data_in_Series already excludes NaN
    from ranking automatically. The table's title is stored in the returned
    TableData's ``df.attrs["title"]``.
    """
    table = TableData.from_DataFrame(raw_df)
    table.format_and_color_cells(color_fn=color_fn, fmt=fmt)
    if emphasis_rankings:
        table.highlight_best_and_worst_results_by_column(higher_is_better=higher_is_better)
    table.df.attrs["title"] = title
    return table


def _style_combined_columns(raw_df_a: pd.DataFrame, raw_df_b: pd.DataFrame, title: str,
                            color_fn=None, fmt=None,
                            highlight: bool = True, higher_is_better: bool = True,
                            separator: str = '/') -> TableData:
    """Resolve two raw numeric DataFrames into one merged-segment TableData.

    Each column's two values (e.g. successful/total counts) are ranked
    independently and combined into a single "A<separator>B" cell. The
    table's title is stored in the returned TableData's ``df.attrs["title"]``.
    """
    table_a = TableData.from_DataFrame(raw_df_a)
    table_a.format_and_color_cells(color_fn=color_fn, fmt=fmt)
    if highlight:
        table_a.highlight_best_and_worst_results_by_column(higher_is_better=higher_is_better)
    table_b = TableData.from_DataFrame(raw_df_b)
    table_b.format_and_color_cells(color_fn=color_fn, fmt=fmt)
    if highlight:
        table_b.highlight_best_and_worst_results_by_column(higher_is_better=higher_is_better)
    merged = TableData.merge_TableData(table_a, table_b, separator=separator)
    merged.df.attrs["title"] = title
    return merged


def pair_label(name_a: str, name_b: str) -> str:
    def abbrev(n):
        m = re.match(r'([A-Za-z]+)(\d+)', n)
        if m:
            return m.group(1)[0].upper() + m.group(2)
        # No trailing number (e.g. "drone", "robotA"): abbreviate to the
        # first letter, keeping a trailing capital if present to distinguish
        # same-base-name robots (e.g. "robotA"/"robotB" -> "RA"/"RB").
        first = n[0].upper()
        if len(n) > 1 and n[-1].isupper():
            return first + n[-1]
        return first
    return abbrev(name_a) + '-' + abbrev(name_b)

class LCFilterMode(Enum):
    """
    Which subset of loop closures to load for a ROMAN run.

    Attributes:
        ALL: Load both inter-robot and intra-robot loop closures.
        ONLY_INTER_LC: Load only inter-robot (cross-robot) loop closures.
        ONLY_INTRA_LC: Load only intra-robot (single-robot) loop closures.
    """

    ALL = 0
    ONLY_INTER_LC = 1
    ONLY_INTRA_LC = 2

@dataclass
class ROMANResults:
    """
    All computed results for one run (method) on one robot pair within a dataset.

    Populated incrementally: :func:`calculate_merged_ate` fills in the four
    trajectory-error fields; ``timing``/``data_size_mb``/``mg_match`` and the
    per-``LCFilterMode`` loop-closure stats are set afterward as each is
    computed in :func:`run_ROMAN_evaluation`.

    Attributes:
        first_stage_metrics: Pre-optimize trajectory error metrics on the merged
            (both-robot) trajectory, or None if the pre-optimize file was unavailable.
        merged_metrics: Post-optimize trajectory error metrics on the merged trajectory.
        robot0_metrics: Post-optimize trajectory error metrics for the first robot,
            computed by separating the merged aligned trajectory back apart.
        robot1_metrics: Post-optimize trajectory error metrics for the second robot.
        timing: Runtime breakdown (``{"align": seconds, "mapping": seconds, "offline_rpgo": seconds}``),
            or None if the runtime files were unavailable.
        data_size_mb: Estimated communication data size in decimal MB, or None
            if the data size file was unavailable.
        mg_match: MG two-stage matcher stage-count/field stats (see
            :func:`load_mg_match_stats_ROMAN`), or None if not an MG run or no
            match files exist.
        lc_stats_by_mode: All-LC stats (see
            :meth:`LoopClosureData.visualize_error_scatter`), keyed by ``LCFilterMode``.
        lc_inlier_stats_by_mode: Inlier-LC stats, keyed by ``LCFilterMode``.
    """
    first_stage_metrics: Optional[PathDataAlignResult]
    merged_metrics: PathDataAlignResult
    robot0_metrics: PathDataAlignResult
    robot1_metrics: PathDataAlignResult
    timing: Optional[Dict[str, float]] = None
    data_size_mb: Optional[float] = None
    mg_match: Optional[Dict] = None
    lc_stats_by_mode: Dict[LCFilterMode, Dict] = field(default_factory=dict)
    lc_inlier_stats_by_mode: Dict[LCFilterMode, Dict] = field(default_factory=dict)

def load_est_data_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str,
                        robot_names: List, critical_invocation_params: Dict[str, Any]) -> List[OdometryData]:
    """
    Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    rpgo_dir = system_params.rpgo_result_dir(Path(roman_root) / "results", dataset_prefix, dataset_name,
                                             sorted(robot_names), critical_invocation_params)
    return [
        OdometryData.from_csv(
            str(rpgo_dir / f'{rn}.csv'),
            "map", 'robot' + str(i), CoordinateFrame.NONE, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        for i, rn in enumerate(robot_names)
    ]

def load_kimera_rpgo_first_stage_est_data_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str,
                                                robot_names: List, critical_invocation_params: Dict[str, Any]) -> List[OdometryData]:
    """
    Load pre-optimize (first-stage) estimated trajectories for a set of robots.

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    sorted_names = sorted(robot_names)
    rpgo_dir = system_params.rpgo_result_dir(Path(roman_root) / "results", dataset_prefix, dataset_name,
                                             sorted_names, critical_invocation_params)
    names_override = {chr(97 + i): name for i, name in enumerate(sorted_names)}
    return [
        OdometryData.from_g2o(str(rpgo_dir / 'pre_optimize' / 'result.g2o'), str(rpgo_dir / 'dense' / 'odom_all.time.txt'), rn,
            "map", 'robot' + str(i), CoordinateFrame.NONE, names_override)
        for i, rn in enumerate(robot_names)
    ]

def load_LC_data_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str, robot_names: List,
                       critical_invocation_params: Dict[str, Any],
                       lc_filter: LCFilterMode = LCFilterMode.ALL, names_override: Dict = None):
    """
    Load LC data for a ROMAN run. names_override, if given, maps g2o character keys ('a', 'b', ...)
    to robot names used in the returned LoopClosureData (default: the sorted robot_names) -- pass
    display-name overrides when LC names must match a visualize_2D nameList.

    Returns:
        (merged_lc, merged_lc_inlier)
    """
    sorted_names = sorted(robot_names)
    rpgo_dir = system_params.rpgo_result_dir(Path(roman_root) / "results", dataset_prefix, dataset_name,
                                             sorted_names, critical_invocation_params)
    letter_by_name = {name: chr(97 + i) for i, name in enumerate(sorted_names)}

    if lc_filter == LCFilterMode.ONLY_INTER_LC:
        pair_fn = itertools.combinations
    elif lc_filter == LCFilterMode.ONLY_INTRA_LC:
        pair_fn = lambda names, num: [(name, name) for name in names]
    else:
        pair_fn = itertools.combinations_with_replacement

    effective_override = names_override if names_override is not None else \
        {chr(97 + i): name for i, name in enumerate(sorted_names)}

    # odom_and_lc.g2o already contains all robot pairs — load it once to avoid
    # tripling the count when iterating over combinations_with_replacement.
    merged_lc = LoopClosureData.from_g2o(
        rpgo_dir / 'dense' / 'odom_and_lc.g2o',
        rpgo_dir / 'dense' / 'odom_all.time.txt',
        names_override=effective_override)
    if lc_filter == LCFilterMode.ONLY_INTER_LC:
        merged_lc.prune_intra_robot_loop_closures()
    elif lc_filter == LCFilterMode.ONLY_INTRA_LC:
        merged_lc.prune_inter_robot_loop_closures()

    # Load the per-pair inlier g2o files (these are pair-specific)
    lc_inlier_data_list = []
    for name_a, name_b in pair_fn(sorted_names, 2):
        letter_a = letter_by_name[name_a]
        letter_b = letter_by_name[name_b]
        if name_a == name_b:
            g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else:
            g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        lc_data_inlier = LoopClosureData.from_g2o(rpgo_dir / g2o_filename,
                                                  rpgo_dir / 'dense' / 'odom_all.time.txt',
                                                  names_override=effective_override)
        lc_inlier_data_list.append(lc_data_inlier)

    merged_lc_inlier = LoopClosureData.merge(lc_inlier_data_list)

    # Get information on duplicates and then prune them
    #merged_lc.print_duplicate_info(f"{dataset_name} {robot_names} ALL")
    #merged_lc_inlier.print_duplicate_info(f"{dataset_name} {robot_names} Kimera-RPGO Inlier")

    merged_lc.prune_duplicates()
    merged_lc_inlier.prune_duplicates()

    return merged_lc, merged_lc_inlier

def load_timing_data_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str,
                           robot_names: List, critical_invocation_params: Dict[str, Any]) -> Dict[str, float]:
    """
    Load the runtime (s) breakdown for a ROMAN run on a robot pair.

    Reads each combination's own alignment runtime (``<robot_a>_<robot_b>.runtime.txt``, one line,
    at that combination's own align result dir), each robot's own mapping runtime
    (``<robot>.runtime.txt``, at its own mapping result dir), and the offline RPGO runtime
    (``runtime.txt``, a single value on its last non-empty line, at the rpgo result dir).

    Returns:
        Dict with keys ``"align"`` (sum of the alignment runtimes),
        ``"mapping"`` (sum of the mapping runtimes), and ``"offline_rpgo"``,
        or ``None`` if any runtime file is missing or empty.
    """
    results_root = Path(roman_root) / "results"

    align_paths = []
    for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):
        sorted_pair = sorted((name_a, name_b))
        align_dir = system_params.align_result_dir(results_root, dataset_prefix, dataset_name,
                                                    sorted_pair[0], sorted_pair[1], critical_invocation_params)
        align_paths.append(align_dir / f'{sorted_pair[0]}_{sorted_pair[1]}.runtime.txt')

    mapping_paths = [
        system_params.mapping_result_dir(results_root, dataset_prefix, dataset_name, rn, critical_invocation_params) / f'{rn}.runtime.txt'
        for rn in robot_names
    ]

    rpgo_runtime_path = system_params.rpgo_result_dir(results_root, dataset_prefix, dataset_name,
                                                       sorted(robot_names), critical_invocation_params) / 'runtime.txt'

    if not all(p.exists() for p in align_paths) or not all(p.exists() for p in mapping_paths) or not rpgo_runtime_path.exists():
        return None

    align_lines = [p.read_text().strip() for p in align_paths]
    mapping_lines = [p.read_text().strip() for p in mapping_paths]
    rpgo_lines = [line.strip() for line in rpgo_runtime_path.read_text().splitlines() if line.strip()]
    if not all(align_lines) or not all(mapping_lines) or not rpgo_lines:
        return None

    align_total = sum(float(line.split(':')[-1]) for line in align_lines)
    mapping_total = sum(float(line.split(':')[-1]) for line in mapping_lines)
    rpgo_total = float(rpgo_lines[-1])

    return {"align": align_total, "mapping": mapping_total, "offline_rpgo": rpgo_total}


def load_data_size_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str,
                         robot_names: List, critical_invocation_params: Dict[str, Any]) -> float:
    """
    Load the estimated communication data size (decimal MB, 1 MB = 1,000,000 bytes)
    for a ROMAN run on a robot pair.

    Reads ``align.data_size.txt`` at that pair's align result dir, a single
    ``"Total submap data size (bytes): <value>"`` line.

    Returns:
        The data size in decimal MB (not MiB), or ``None`` if the file is missing.
    """
    align_dir = system_params.align_result_dir(Path(roman_root) / "results", dataset_prefix, dataset_name,
                                                robot_names[0], robot_names[1], critical_invocation_params)
    data_size_path = align_dir / 'align.data_size.txt'
    if not data_size_path.exists():
        return None

    line = data_size_path.read_text().strip()
    data_size_bytes = float(line.split(':')[-1])
    return data_size_bytes / 1_000_000


def load_mg_match_stats_ROMAN(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str,
                              robot_names: List, critical_invocation_params: Dict[str, Any]) -> Dict:
    """
    Count MG two-stage matcher calls by stage for a robot pair.

    Reads ``align.mg_match.txt`` for each of the (up to three) intra-/inter-robot combinations,
    at each combination's own align result dir, and tallies how many calls reached stage 0, 1, or
    2. Also collects, across all stage-1/2 calls, the values of ``n_stage1_matches`` and, across
    all stage-2 calls, ``n_stage2_child_clipper``,
    ``n_stage2_unmatched_children_to_parents_clipper``,
    ``n_stage2_unmatched_children_to_children_clipper``, and
    ``stage2_point_error``. Fields absent from a given line (older log
    formats don't include all fields) are skipped for that line rather than
    raising. Files that don't exist (e.g. non-MG methods) are skipped.

    Returns:
        Dict with ``"stage_counts"`` (stage -> occurrence count) and one
        list of values per collected field name above (``stage2_point_error``
        as floats, possibly ``nan``; the rest as ints), or ``None`` if no
        ``align.mg_match.txt`` files were found for this pair.
    """
    results_root = Path(roman_root) / "results"

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
        align_dir = system_params.align_result_dir(results_root, dataset_prefix, dataset_name,
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


def calculate_merged_ate(roman_root: Path, system_params, dataset_prefix: str, dataset_name: str, method: str, robot_names: List,
                         critical_invocation_params: Dict[str, Any], load_gt_data_fn,
                         figures_base_dir: Path = None, visualize: bool = False, do_individual_calcs: bool = False,
                         viz_config: Dict = None,
                         rpe_delta: float = 5.0, rpe_delta_unit: Unit = Unit.meters) -> ROMANResults:
    """
    Compute the merged RMS ATE for a robot pair after ROMAN offline RPGO.

    Concatenates the estimated and ground-truth trajectories for both robots
    (after aligning their time windows), then computes ATE on the combined
    trajectory. Optionally also computes the pre-optimize (first-stage) ATE
    and saves trajectory / LC overlay plots.

    Args:
        roman_root: Path to the roman repo checkout.
        system_params: From load_system_params_ROMAN, for this dataset_prefix/dataset_name/method.
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``), forwarded to :func:`load_est_data_ROMAN`,
            :func:`load_kimera_rpgo_first_stage_est_data_ROMAN`, and :func:`load_LC_data_ROMAN`.
        dataset_name: Dataset identifier (e.g. ``"V2.3.AC"``).
        method: Run name, used for figure/file naming only (system_params already resolves the
            actual result directories).
        robot_names: Two-element list of robot names (e.g. ``["Husky1", "Drone1"]``).
        critical_invocation_params: Other data-affecting args from the original run invocation.
        load_gt_data_fn: Callable ``(dataset_name, robot_names) -> List[OdometryData]``,
            dataset-specific.
        figures_base_dir: Directory under which ``<dataset_prefix>/<dataset_name>/traj`` and
            ``<dataset_prefix>/<dataset_name>/<LCFilterMode.name>/traj_lc`` outputs are saved.
            Required when ``visualize`` is True.
        visualize: If True, generate and save 2D trajectory and LC overlay PDFs. Requires
            ``figures_base_dir`` and ``viz_config``.
        do_individual_calcs: If True, also print per-robot ATE before the merged calc.
        viz_config: Dict with keys ``"image_path"``, ``"x_edge"``, ``"robot_name_to_color"``
            (keyed by display name), and optionally ``"name_map"`` (robot name -> display name;
            defaults to identity) and ``"yaw_rotation_deg"`` (rotates trajectories about the
            center of their combined bounding box before plotting against the background
            image; defaults to 0). Required when ``visualize`` is True.
        rpe_delta: Step size between the pose pairs used for all RPE calculations in
            this function (first-stage, merged, and individual). Does not affect ATE.
            Defaults to ``5.0``.
        rpe_delta_unit: Unit of ``rpe_delta``. Defaults to ``Unit.meters``, matching
            the fixed-distance RPE convention used by GrAco.

    Returns:
        A :class:`DatasetSequenceResults` with ``first_stage_metrics``,
        ``merged_metrics``, ``robot0_metrics``, and ``robot1_metrics`` filled in
        (all other fields default and are filled in later by the caller).
        ``first_stage_metrics`` is the pre-optimize :class:`PathDataAlignResult`
        (``None`` if the pre-optimize file is unavailable); ``merged_metrics`` is the
        :class:`PathDataAlignResult` computed on the merged (both-robot),
        post-optimize trajectory (e.g. ``merged_metrics.APE.translation_part.rmse``
        for RMS ATE); ``robot0_metrics``/``robot1_metrics`` are each robot's
        post-optimize :class:`PathDataAlignResult`, computed by separating the
        merged aligned trajectories back into per-robot trajectories (via
        :meth:`PathData.seperate_PathData`) and calling
        :meth:`PathData.calculate_traj_errors` on each robot's already-aligned pair.

    Raises:
        ValueError: If ``robot_names`` does not have exactly two entries.
    """
    if len(robot_names) != 2:
        raise ValueError(f"robot_names must have exactly two entries, got {len(robot_names)}: {robot_names}")
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]

    est_data_lst: List[OdometryData] = load_est_data_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                           robot_names, critical_invocation_params)
    gt_data_lst: List[OdometryData] = load_gt_data_fn(dataset_name, robot_names)
    est_data_robot0, est_data_robot1 = est_data_lst
    gt_data_robot0, gt_data_robot1 = gt_data_lst

    # Calculate individual RMS ATE without aligning
    if do_individual_calcs:
        # TODO: Need to make start and end times match before individual RMS ATE as well;
        # if we ever use those results in a paper.
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False,
                                                                                rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False,
                                                                                rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)

    # Calculate first-stage (pre-optimize) metrics
    first_stage_metrics = None
    try:
        first_stage_est_lst = load_kimera_rpgo_first_stage_est_data_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                                          robot_names, critical_invocation_params)
        first_stage_est_lst, first_stage_gt_lst = PathData.make_start_and_end_times_match(first_stage_est_lst, gt_data_lst)
        first_stage_est: PathData = PathData.concatenate_PathData(first_stage_est_lst)
        first_stage_gt: PathData = PathData.concatenate_PathData(first_stage_gt_lst)
        #print("\n========== First Stage for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
        first_stage_metrics, _, _ = OdometryData.align_and_calculate_traj_errors(first_stage_gt, first_stage_est, max_diff=0.1, visualize=False,
                                                                                 rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
    except Exception as e:
        print(f"Warning: Could not compute first-stage metrics for {dataset_name} {method}: {e}")

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    #print("\n========== Merged Trajectories for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False,
                                                                                                     rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)

    # Seperate the aligned trajectories into their single-robot forms, and compute
    # each robot's individual post-optimize RMS ATE from its already-aligned pair.
    gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
    gt_data_align_robot0 = gt_data_align_list[0]
    gt_data_align_robot1 = gt_data_align_list[1]

    est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
    est_data_align_robot0 = est_data_align_list[0]
    est_data_align_robot1 = est_data_align_list[1]

    robot0_metrics, robot1_metrics = [
        PathData.calculate_traj_errors(gt_align, est_align, rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        for gt_align, est_align in zip(gt_data_align_list, est_data_align_list)
    ]

    if visualize:
        image_path = viz_config["image_path"]
        x_edge = viz_config["x_edge"]
        name_map: Dict = viz_config.get("name_map") or {rn: rn for rn in robot_names}
        robot_name_to_color: Dict = viz_config["robot_name_to_color"]
        image_extent_offsets = viz_config.get("background_image_extent_offsets")
        yaw_rotation_deg = viz_config.get("yaw_rotation_deg", 0.0)

        pair_lbl = pair_label(robot0_name, robot1_name)
        base_dir = Path(figures_base_dir) / dataset_prefix / dataset_name
        traj_dir = base_dir / 'traj'
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Plot the results in 2D (Configuration for Figure 10) — LC-independent, saved once
        dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
        isGTList =  [                False,                 True,                  False,                  True]
        nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True,
                           background_image_path=image_path, background_image_x_edge=x_edge,
                           background_image_extent_offsets=image_extent_offsets,
                           yaw_rotation_deg=yaw_rotation_deg,
                           save_path=str(traj_dir / f'traj_{pair_lbl}_{method}.pdf'))

        # Plot only GT in 2D
        dataList =  [gt_data_align_robot0,  gt_data_align_robot1]
        isGTList =  [                True,                  True]
        nameList =  [name_map[robot0_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=False,
                           background_image_path=image_path, background_image_x_edge=x_edge,
                           background_image_extent_offsets=image_extent_offsets,
                           gt_color_lightness_range_val=8,
                           yaw_rotation_deg=yaw_rotation_deg,
                           save_path=str(traj_dir / f'traj_{pair_lbl}_{method}_onlyGT.pdf'))

        # Plot estimated trajectories with LC overlay (no background, no GT), once per LC filter mode.
        # Letters are assigned by sorted robot order to match the g2o files' own convention.
        names_override_display = {chr(97 + i): name_map[rn] for i, rn in enumerate(sorted([robot0_name, robot1_name]))}
        gt_dict_display = {name_map[robot0_name]: gt_data_robot0, name_map[robot1_name]: gt_data_robot1}
        est_dataList =  [est_data_align_robot0,       est_data_align_robot1]
        est_isGTList =  [               False,                        False]
        est_nameList =  [name_map[robot0_name], name_map[robot1_name]]
        est_colorList = [robot_name_to_color[name] for name in est_nameList]
        for lc_filter in LCFilterMode:
            _, lc_data_inlier = load_LC_data_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                   [robot0_name, robot1_name], critical_invocation_params,
                                                   lc_filter=lc_filter, names_override=names_override_display)
            lc_data_inlier.calculate_errors(gt_dict_display)

            traj_lc_dir = base_dir / lc_filter.name / 'traj_lc'
            traj_lc_dir.mkdir(parents=True, exist_ok=True)
            PathData.visualize_2D(est_dataList, est_isGTList, est_colorList, est_nameList, no_background=True, line_width=1.0, show_grid=True,
                               loop_closure_data=lc_data_inlier, lc_line_width=2.0, lc_errors_vmax=2.0,
                               title=f"{method} LC overlaid on trajectory",
                               save_path=str(traj_lc_dir / f'traj_lc_{pair_lbl}_{method}.pdf'))

    return ROMANResults(first_stage_metrics, metrics_dictionary, robot0_metrics, robot1_metrics)


def _save_ate_tables(run_names: List[str], cols: List[str],
                     run_display_names: Dict[str, str],
                     results: Dict[str, Dict[str, ROMANResults]],
                     save_path: Path, ate_threshold_m: float, rot_threshold_deg: float) -> None:
    """
    Build and save the ATE/RTE summary PDF tables.

    Produces five tables — pre-optimize (first-stage) merged RMS ATE,
    post-optimize merged RMS ATE, post-optimize merged RMS absolute rotation
    angle error, post-optimize merged RMS RTE, and post-optimize merged RMS
    relative rotation angle error — styled so that cells with no loop
    closures, or a value above ate_threshold_m (translation tables) / above
    rot_threshold_deg (rotation tables), are highlighted in red. The
    pre-optimize table is suppressed for pairs with zero total inter-robot LC;
    the post-optimize ATE, rotation error, RTE, and relative rotation error
    tables for pairs with zero inlier inter-robot LC (both via
    ``results[...].lc_stats_by_mode``/``lc_inlier_stats_by_mode`` at
    ``LCFilterMode.ONLY_INTER_LC``).

    Each table gets a trailing "Average" column (the row-wise mean across
    the pair columns, ignoring suppressed/NaN pairs), set off from the pair
    columns by a heavy divider.

    Args:
        run_names: Ordered list of run identifiers (e.g. ``["ROMAN", "MG_TS"]``).
        cols: Ordered list of robot-pair column labels (e.g. ``["H1H2", "H1D1"]``).
        run_display_names: Maps each run identifier to its display name in the table.
        results: ``DatasetSequenceResults`` keyed by run then column.
        save_path: Destination PDF path.
        ate_threshold_m: Red-highlight cutoff (m) for the translation-error tables
            (ATE pre/post-optimize, RTE). Dataset-specific (e.g. smaller for a
            smaller-area dataset like AirMuseum than for Hercules).
        rot_threshold_deg: Red-highlight cutoff (deg) for the rotation-error tables
            (absolute and relative).
    """
    def make_raw_df(metric_fn, lc_stats_selector) -> pd.DataFrame:
        def value_fn(run, col):
            result = results[run].get(col)
            val = metric_fn(result) if result is not None else None
            lc_stats = lc_stats_selector(result) if result is not None else None
            suppressed = val is None or (lc_stats or {}).get('num_loop_closures', -1) == 0
            return float('nan') if suppressed else val
        raw_df = _make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
        raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
        return raw_df

    inter_lc = lambda r: r.lc_stats_by_mode.get(LCFilterMode.ONLY_INTER_LC)
    inter_lc_inlier = lambda r: r.lc_inlier_stats_by_mode.get(LCFilterMode.ONLY_INTER_LC)

    color_fn_m = TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m)
    color_fn_deg = TableData.color_fn_NAVY_RED_missing_or_above(rot_threshold_deg)
    fmt = TableData.fmt_fixed(3)
    # The trailing "Average" column is a summary column, not another pair —
    # set it off from the pair columns with a heavy divider.
    heavy_divider_before = lambda col_idx: col_idx == len(cols)

    first_stage_ate_table = make_highlighted_table(
                   make_raw_df(lambda r: r.first_stage_metrics.APE.translation_part.rmse if r.first_stage_metrics else None, inter_lc),
                   "Merged RMS ATE (m) — Pre-Optimize",
                   color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
    ate_table = make_highlighted_table(
                   make_raw_df(lambda r: r.merged_metrics.APE.translation_part.rmse, inter_lc_inlier),
                   "Merged RMS ATE (m)",
                   color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
    rot_err_table = make_highlighted_table(
                   make_raw_df(lambda r: r.merged_metrics.APE.rotation_angle_deg.rmse, inter_lc_inlier),
                   "Merged RMS Absolute Rotation Error (deg)",
                   color_fn=color_fn_deg, fmt=fmt, higher_is_better=False)
    rte_table = make_highlighted_table(
                   make_raw_df(lambda r: r.merged_metrics.RPE.translation_part.rmse, inter_lc_inlier),
                   "Merged RMS RTE (m) - Δ5m",
                   color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
    rel_rot_err_table = make_highlighted_table(
                   make_raw_df(lambda r: r.merged_metrics.RPE.rotation_angle_deg.rmse, inter_lc_inlier),
                   "Merged RMS Relative Rotation Error (deg) - Δ5m",
                   color_fn=color_fn_deg, fmt=fmt, higher_is_better=False)

    dfs = [first_stage_ate_table, ate_table, rot_err_table, rte_table, rel_rot_err_table]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, heavy_divider_before=heavy_divider_before)

    color_fn_latex = TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m, style=TableData.TableStyleName.LATEX)
    ate_table.format_and_color_cells(color_fn=color_fn_latex, fmt=fmt)
    ate_table.highlight_best_and_worst_results_by_column(higher_is_better=False, rank_styles=[TableData.TextStyle.BOLD])
    ate_table.set_title("Method")
    ate_table.to_latex(str(save_path.parent / f"{save_path.stem}_ate.tex"),
        caption="RMS ATE (m).",
        label="tab:merged_rms_ate")


def _save_ate_split_table(run_names: List[str], robot_pairs: List[Tuple[str, str]],
                          run_display_names: Dict[str, str],
                          results: Dict[str, Dict[str, ROMANResults]],
                          save_path: Path, ate_threshold_m: float) -> None:
    """
    Build and save the per-robot RMS ATE/RPE split summary PDF tables.

    For each robot pair, produces two adjacent columns (one per robot) holding
    that robot's individual post-optimize RMS ATE/RPE, computed by separating
    the merged aligned trajectory back into per-robot trajectories (see
    :func:`calculate_merged_ate`).

    Args:
        run_names: Ordered list of run identifiers.
        robot_pairs: Ordered list of two-robot-name tuples, matching the pair
            order used to build ``results``.
        run_display_names: Maps each run identifier to its display name in the table.
        results: ``DatasetSequenceResults`` keyed by run then column.
        save_path: Destination PDF path.
        ate_threshold_m: Red-highlight cutoff (m) for both tables (both are translation-only).
    """
    sub_cols = [f"{pair_label(a, b)}\n{name}" for a, b in robot_pairs for name in (a, b)]

    subcol_pair_idx = {f"{pair_label(a, b)}\n{name}": (pair_label(a, b), i)
                      for a, b in robot_pairs for i, name in enumerate((a, b))}

    def make_raw_df(metric_fn) -> pd.DataFrame:
        def value_fn(run, subcol):
            pair, i = subcol_pair_idx[subcol]
            result = results[run].get(pair)
            robot_metrics = (result.robot1_metrics if i == 1 else result.robot0_metrics) if result is not None else None
            return float('nan') if robot_metrics is None else metric_fn(robot_metrics)
        return _make_raw_df(run_names, run_display_names, lambda run: sub_cols, value_fn)

    color_fn = TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m)
    fmt = TableData.fmt_fixed(3)

    dfs = [
        make_highlighted_table(make_raw_df(lambda m: m.APE.translation_part.rmse), "Individual RMS ATE (m)",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
        make_highlighted_table(make_raw_df(lambda m: m.RPE.translation_part.rmse), "Individual RMS RPE (m) - Δ5m",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5)


def _save_timing_table(run_names: List[str], cols: List[str],
                       run_display_names: Dict[str, str],
                       results: Dict[str, Dict[str, ROMANResults]],
                       save_path: Path) -> None:
    """
    Build and save the runtime summary PDF table.

    Produces one table per run and robot pair for each of the alignment
    runtime, the mapping runtime, the offline RPGO runtime, and their
    per-pair total.

    Args:
        run_names: Ordered list of run identifiers.
        cols: Ordered list of robot-pair column labels.
        run_display_names: Maps each run identifier to its display name in the table.
        results: ``DatasetSequenceResults`` keyed by run then column; ``.timing``
            is ``None`` for pairs with unavailable runtime files.
        save_path: Destination PDF path.
    """
    def get_val(run: str, col: str, key: str):
        result = results[run].get(col)
        entry = result.timing if result is not None else None
        if key == "total":
            return None if entry is None else entry["align"] + entry["mapping"] + entry["offline_rpgo"]
        return None if entry is None else entry[key]

    def make_raw_df(key: str) -> pd.DataFrame:
        def value_fn(run, col):
            v = get_val(run, col, key)
            return float('nan') if v is None else v
        raw_df = _make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
        raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
        return raw_df

    style = TableData.TableStyleName.GEORGIA_TECH
    color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'), style=style)
    fmt = TableData.fmt_fixed(1)
    # The trailing "Average" column is a summary column, not another pair —
    # set it off from the pair columns with a heavy divider.
    heavy_divider_before = lambda col_idx: col_idx == len(cols)

    dfs = [
        make_highlighted_table(make_raw_df("mapping"), "Mapping Runtime (s)",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
        make_highlighted_table(make_raw_df("align"), "Alignment Runtime (s)",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
        make_highlighted_table(make_raw_df("offline_rpgo"), "Offline RPGO Runtime (s)",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
        make_highlighted_table(make_raw_df("total"), "Total Runtime (s)",
                       color_fn=color_fn, fmt=fmt, higher_is_better=False),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, style=style,
                      heavy_divider_before=heavy_divider_before)


def _save_data_size_table(run_names: List[str], cols: List[str],
                          run_display_names: Dict[str, str],
                          results: Dict[str, Dict[str, ROMANResults]],
                          save_path: Path) -> None:
    """
    Build and save the estimated communication data size summary PDF table.

    Also saves a standalone ``.tex`` version of the table (same path with a
    ``.tex`` suffix), ready to paste into Overleaf.

    Args:
        run_names: Ordered list of run identifiers.
        cols: Ordered list of robot-pair column labels.
        run_display_names: Maps each run identifier to its display name in the table.
        results: ``DatasetSequenceResults`` keyed by run then column; ``.data_size_mb``
            is ``None`` for pairs with unavailable data size files.
        save_path: Destination PDF path.
    """
    def make_raw_df() -> pd.DataFrame:
        def value_fn(run, col):
            result = results[run].get(col)
            v = result.data_size_mb if result is not None else None
            return float('nan') if v is None else v
        raw_df = _make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
        raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
        return raw_df

    style = TableData.TableStyleName.GEORGIA_TECH
    color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'), style=style)
    fmt = TableData.fmt_fixed(2)
    # The trailing "Average" column is a summary column, not another pair —
    # set it off from the pair columns with a heavy divider.
    heavy_divider_before = lambda col_idx: col_idx == len(cols)

    data_size_table = make_highlighted_table(make_raw_df(), "Estimated Communication Data Size (MB)",
                         color_fn=color_fn, fmt=fmt, higher_is_better=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    TableData.to_pdf([data_size_table], str(save_path), row_height=2.4, h_pad=0.5, style=style,
                      heavy_divider_before=heavy_divider_before)
    data_size_table.to_latex(str(save_path.with_suffix('.tex')),
                              caption="Estimated Communication Data Size (MB)", label="tab:data_size")


def _save_lc_tables(run_names: List[str], run_display_names: Dict[str, str],
                    results: Dict[str, Dict[str, ROMANResults]],
                    lc_filter: LCFilterMode,
                    save_path: Path) -> None:
    """
    Build and save the LC summary PDF tables.

    Produces four tables: success rate and successful/total counts for all LC
    and for inlier LC, across all run names and robot pairs. The all-LC
    success rate table gets a trailing "Average" column (the row-wise mean
    across the pair columns, ignoring suppressed/NaN pairs), set off from the
    pair columns by a heavy divider — matching ``_save_ate_tables``.

    When ``lc_filter`` is ``LCFilterMode.ALL``, the all-LC success rate and
    successful/total tables are additionally saved as standalone ``.tex``
    files (``lc_success_rate_table.tex``, ``lc_successful_total_table.tex``)
    next to ``save_path``, ready to paste into Overleaf.

    Args:
        run_names: Ordered list of run identifiers.
        run_display_names: Maps each run identifier to its display name in the table.
        results: ``DatasetSequenceResults`` keyed by run then column; stats are
            read from ``.lc_stats_by_mode``/``.lc_inlier_stats_by_mode`` at ``lc_filter``.
        lc_filter: Which ``LCFilterMode`` to pull stats for.
        save_path: Destination PDF path.
    """
    def make_raw_df(stats_selector, key: str) -> pd.DataFrame:
        return _make_raw_df(run_names, run_display_names, lambda run: results[run].keys(),
                            lambda run, col: float(stats_selector(results[run][col])[key]))

    def make_success_rate_df(stats_selector) -> pd.DataFrame:
        raw_df = make_raw_df(stats_selector, "success_rate")
        raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
        return raw_df

    all_lc = lambda r: r.lc_stats_by_mode[lc_filter]
    inlier_lc = lambda r: r.lc_inlier_stats_by_mode[lc_filter]

    percent_fmt = TableData.fmt_fixed(1, suffix='%')
    int_fmt = TableData.fmt_fixed(0)
    color_fn = TableData.color_fn_NAVY_RED_missing_or_equal(style=TableData.TableStyleName.GEORGIA_TECH)
    latex_color_fn = TableData.color_fn_NAVY_RED_missing_or_equal(style=TableData.TableStyleName.LATEX)
    # The trailing "Average" column on the success rate tables is a summary
    # column, not another pair — set it off from the pair columns with a
    # heavy divider (matches _save_ate_tables).
    heavy_divider_before = lambda col_idx: col_idx == len(list(results[run_names[0]].keys()))

    all_lc_success_rate_table = make_highlighted_table(make_success_rate_df(all_lc), "LC Success Rate %",
                   color_fn=color_fn, fmt=percent_fmt)
    all_lc_successful_total_table = _style_combined_columns(make_raw_df(all_lc, "num_successful_loop_closures"),
                            make_raw_df(all_lc, "num_loop_closures"),
                            "LC Successful / Total", color_fn=color_fn, fmt=int_fmt)

    all_lc_success_rate_table_latex = make_highlighted_table(make_success_rate_df(all_lc), "LC Success Rate %",
                   color_fn=latex_color_fn, fmt=percent_fmt)
    all_lc_successful_total_table_latex = _style_combined_columns(make_raw_df(all_lc, "num_successful_loop_closures"),
                            make_raw_df(all_lc, "num_loop_closures"),
                            "LC Successful / Total", color_fn=latex_color_fn, fmt=int_fmt)

    dfs = [
        all_lc_success_rate_table,
        all_lc_successful_total_table,
        make_highlighted_table(make_raw_df(inlier_lc, "success_rate"), "Inlier LC Success Rate %",
                       color_fn=color_fn, fmt=percent_fmt),
        _style_combined_columns(make_raw_df(inlier_lc, "num_successful_loop_closures"),
                                make_raw_df(inlier_lc, "num_loop_closures"),
                                "Inlier LC Successful / Total", color_fn=color_fn, fmt=int_fmt),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, style=TableData.TableStyleName.GEORGIA_TECH,
                     heavy_divider_before=heavy_divider_before)

    if lc_filter == LCFilterMode.ALL:
        all_lc_success_rate_table_latex.to_latex(str(save_path.parent / 'lc_success_rate_table.tex'),
                                            caption="LC Success Rate \%", label="tab:lc_success_rate")
        all_lc_successful_total_table_latex.to_latex(str(save_path.parent / 'lc_successful_total_table.tex'),
                                                caption="LC Successful / Total", label="tab:lc_successful_total")


def _make_mg_match_histogram_grid_figure(run_names: List[str], run_display_names: Dict[str, str],
                                         cols: List[str], table_data_mg_match: Dict[str, Dict[str, Dict]],
                                         field: str, title: str, discrete: bool = True):
    """Build a grid-of-histograms figure for one MG match stat field.

    Lays out one small histogram per (method, robot pair) cell, rows=methods,
    cols=robot pairs, so per-call value distributions can be inspected
    directly instead of collapsing them to a mean/std. For discrete fields,
    bins are width-1 and anchored on half-integers (``-0.5, 0.5, 1.5, ...``);
    for continuous fields, 10 evenly spaced bins span the field's global
    range. NaNs (e.g. unset ``stage2_point_error``) are dropped.
    """
    def clean(values):
        return [v for v in values if not pd.isna(v)]

    active_run_names = [run for run in run_names
                        if any(table_data_mg_match[run].get(col) is not None for col in cols)]

    if not active_run_names:
        fig, ax = plt.subplots(1, 1, figsize=(3.0 * len(cols), 2.2))
        ax.text(0.5, 0.5, "No MG match data", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle(title, fontsize=14, fontweight='bold', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    fig, axes = plt.subplots(len(active_run_names), len(cols), squeeze=False,
                             figsize=(3.0 * len(cols), 2.2 * len(active_run_names)))
    cmap = plt.cm.viridis

    all_values = [v for run in active_run_names for col in cols
                  for v in clean((table_data_mg_match[run].get(col) or {}).get(field) or [])]
    if not all_values:
        bin_edges = None
    elif discrete:
        bin_edges = np.arange(min(all_values) - 0.5, max(all_values) + 1.5, 1)
    else:
        vmin, vmax = min(all_values), max(all_values)
        bin_edges = np.linspace(vmin, vmax, 11) if vmin != vmax else np.array([vmin - 0.5, vmin + 0.5])
    bin_width = (bin_edges[1] - bin_edges[0]) if bin_edges is not None else None

    max_count = 0
    if bin_edges is not None:
        for run in active_run_names:
            for col in cols:
                stats = table_data_mg_match[run].get(col)
                values = clean(stats[field]) if stats is not None else []
                if values:
                    max_count = max(max_count, np.histogram(values, bins=bin_edges)[0].max())

    for i, run in enumerate(active_run_names):
        for j, col in enumerate(cols):
            ax = axes[i][j]
            stats = table_data_mg_match[run].get(col)
            values = clean(stats[field]) if stats is not None else []
            if values:
                counts, edges = np.histogram(values, bins=bin_edges)
                bar_colors = cmap(np.linspace(0.15, 0.85, len(counts)))
                ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=bin_width,
                      color=bar_colors, edgecolor='white', linewidth=0.3)
                ax.set_xlim(bin_edges[0], bin_edges[-1])
                ax.set_ylim(0, max_count * 1.05)
                ax.tick_params(axis='both', labelsize=6)
            else:
                ax.text(0.5, 0.5, "---", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 0:
                ax.set_title(col, fontsize=9)
            if j == 0:
                ax.set_ylabel(run_display_names.get(run, run), fontsize=9, rotation=0, ha='right', va='center')

    fig.suptitle(title, fontsize=14, fontweight='bold', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _save_mg_match_table(run_names: List[str], run_display_names: Dict[str, str], cols: List[str],
                         results: Dict[str, Dict[str, ROMANResults]],
                         save_path: Path) -> None:
    """
    Build and save the MG two-stage matcher stats summary PDF.

    Page 1 holds two styled tables — raw ``stage1-stage2`` call counts, and
    each stage's percentage of that pair's total calls (including stage 0).
    Subsequent pages hold a grid of mini histograms (rows=methods,
    cols=robot pairs) for each of ``n_stage1_matches``,
    ``n_stage2_child_clipper``, ``n_stage2_unmatched_children_to_parents_clipper``,
    ``n_stage2_unmatched_children_to_children_clipper``, and
    ``stage2_point_error``, so the full per-call distribution is visible
    instead of a collapsed mean/std. Runs with no ``align.mg_match.txt``
    files are omitted from the histogram grids entirely; runs/pairs with no
    data for the two summary tables are shown as ``"---"``.

    Args:
        run_names: Ordered list of run identifiers.
        run_display_names: Maps each run identifier to its display name in the table.
        cols: Ordered list of robot-pair column labels.
        results: ``DatasetSequenceResults`` keyed by run then column; ``.mg_match``
            is ``None`` for pairs with no MG match files.
        save_path: Destination PDF path.
    """
    table_data_mg_match: Dict[str, Dict[str, Optional[Dict]]] = {
        run: {col: results[run][col].mg_match for col in results[run]} for run in run_names
    }

    def make_raw_stage_count_df(stage: int) -> pd.DataFrame:
        def value_fn(run, col):
            stats = table_data_mg_match[run][col]
            return float('nan') if stats is None else float(stats['stage_counts'][stage])
        return _make_raw_df(run_names, run_display_names, lambda run: table_data_mg_match[run].keys(), value_fn)

    def make_raw_stage_percent_df(stage: int) -> pd.DataFrame:
        def value_fn(run, col):
            stats = table_data_mg_match[run][col]
            if stats is None:
                return float('nan')
            counts = stats['stage_counts']
            total = counts[0] + counts[1] + counts[2]
            return float('nan') if total == 0 else 100 * counts[stage] / total
        return _make_raw_df(run_names, run_display_names, lambda run: table_data_mg_match[run].keys(), value_fn)

    color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'))
    int_fmt = TableData.fmt_fixed(0)
    percent_fmt = TableData.fmt_fixed(1, suffix='%')

    counts_table = _style_combined_columns(make_raw_stage_count_df(1), make_raw_stage_count_df(2),
                                        "MG Match Stage Counts (1-2)",
                                        color_fn=color_fn, fmt=int_fmt, highlight=False, separator='-')
    percent_table = _style_combined_columns(make_raw_stage_percent_df(1), make_raw_stage_percent_df(2),
                                         "MG Match Stage Percentages (1-2)",
                                         color_fn=color_fn, fmt=percent_fmt, highlight=False, separator='-')

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(save_path)) as pp:
        fig, axes = plt.subplots(2, 1, figsize=(12, 2.4 * 2))
        for ax in axes:
            ax.axis('off')
        fig.tight_layout(pad=0.0, h_pad=0.5)
        counts_table.render_onto_ax(fig, axes[0])
        percent_table.render_onto_ax(fig, axes[1])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        histogram_fields = [
            ("n_stage1_matches", "n_stage1_matches", True),
            ("n_stage2_child_clipper", "Stage 2 Matched C-C (M)", True),
            ("n_stage2_unmatched_children_to_parents_clipper", "Stage 2 Matched C-P (UM)", True),
            ("n_stage2_unmatched_children_to_children_clipper", "Stage 2 Matched C-C (UM)", True),
            ("stage2_point_error", "Stage 2 Point Error", False),
        ]
        for field, title, discrete in histogram_fields:
            hist_fig = _make_mg_match_histogram_grid_figure(run_names, run_display_names, cols,
                                                            table_data_mg_match, field, title, discrete)
            pp.savefig(hist_fig, bbox_inches='tight')
            plt.close(hist_fig)

def _generate_lc_context_figure(pair: Tuple[str, str], col: str,
                                 lc_data_list: List[LoopClosureData], labels_list: List[str],
                                 group_indices: List[int],
                                 stats_list: List[Dict],
                                 results: Dict[str, Dict[str, ROMANResults]],
                                 run_names: List[str], save_dir: Path, ate_threshold_m: float) -> None:
    """
    Generate and save a composite 16:9 slide figure for one robot pair.

    The figure is laid out as a PowerPoint-sized (13.33 × 7.5 in) slide with
    three styled tables (gold headers, alternating rows, bold/italic+underline
    ranking) and a center LC error scatter plot:

      - **Top-left**: pair name label (golden).
      - **Left**: "After Alignment" — combined all-LC table with columns
        ``"LC Success Rate %"`` and ``"LC Successful / Total"`` per run.
      - **Center**: LC error scatter plot (log-log).
      - **Right**: ``"RMS ATE (m)"`` table (top) followed by "After Kimera-RPGO"
        — combined inlier-LC table with ``"Inlier LC Success Rate %"`` and
        ``"Inlier LC Successful / Total"`` per run.

    Table styling is applied via :meth:`TableData.render_onto_ax`.  Column names
    match the PDF table titles produced by :func:`_save_ate_tables` and
    :func:`_save_lc_tables`.

    Args:
        pair: Two robot names identifying this pair (e.g. ``("Husky1", "Drone1")``).
        col: Short pair label used as the table column header and in the filename
            (e.g. ``"H1D1"``).
        lc_data_list: Interleaved list of all-LC and inlier-LC LoopClosureData for
            each run (length ``2 * len(run_names)``), each with ``calculate_errors``
            and ``label_successful`` already called.
        labels_list: Display label for each entry in ``lc_data_list``.
        group_indices: Group index for each entry, pairing all-LC and inlier-LC
            entries within the same run.  Must follow the pattern
            ``[0, 0, 1, 1, ..., n-1, n-1]``.
        stats_list: Interleaved per-run LC stats dicts as returned by
            :meth:`LoopClosureData.visualize_error_scatter` (length
            ``2 * len(run_names)``).  Even indices are all-LC; odd are inlier-LC.
        results: ``DatasetSequenceResults`` keyed by run then column; the ATE
            cell is suppressed where ``results[...].lc_inlier_stats_by_mode``
            at ``LCFilterMode.ONLY_INTER_LC`` has zero loop closures, matching
            :func:`_save_ate_tables`.
        run_names: Ordered list of run identifiers.
        save_dir: Directory in which to save ``lc_context_<col>.pdf``.
        ate_threshold_m: Red-highlight cutoff (m) for the ATE table, matching :func:`_save_ate_tables`.
    """
    expected_group_indices = [i for i in range(len(run_names)) for _ in range(2)]
    assert group_indices == expected_group_indices, (
        f"group_indices must be interleaved pairs [0,0,1,1,...], "
        f"got {group_indices}, expected {expected_group_indices}")

    fig = plt.figure(figsize=(22, 12))

    # Outer: left table | scatter | right column
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.4, 2.2, 1.4],
                           left=0.06, right=0.97, bottom=0.08, top=0.95, wspace=0.18)

    # Left: After Alignment (top) | After Kimera-RPGO (bottom)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0], height_ratios=[1, 1], hspace=0.05)
    ax_l = fig.add_subplot(gs_left[0])
    ax_r = fig.add_subplot(gs_left[1])
    ax_l.axis('off')
    ax_r.axis('off')

    # Center: scatter (forced square box regardless of figure proportions)
    ax_center = fig.add_subplot(gs[1])
    ax_center.set_box_aspect(1)

    # Right: ATE table centered by itself
    ax_ate = fig.add_subplot(gs[2])
    ax_ate.axis('off')

    # Column name constants matching the PDF table titles
    COL_SR_ALL     = "LC Success \n Rate %"
    COL_CNT_ALL    = "LC Successful \n/ Total"
    COL_SR_INL     = "Inlier LC \nSuccess Rate %"
    COL_CNT_INL    = "Inlier LC \nSuccessful / Total"
    COL_ATE        = "RMS ATE (m)"

    percent_fmt = TableData.fmt_fixed(1, suffix='%')
    int_fmt = TableData.fmt_fixed(0)

    def sr_series(idx: int) -> pd.Series:
        return pd.Series(
            {rn: stats_list[2 * i + idx]['success_rate'] for i, rn in enumerate(run_names)}
        ).reindex(run_names)

    def successful_series(idx: int) -> pd.Series:
        return pd.Series(
            {rn: float(stats_list[2 * i + idx]['num_successful_loop_closures']) for i, rn in enumerate(run_names)}
        ).reindex(run_names)

    def total_series(idx: int) -> pd.Series:
        return pd.Series(
            {rn: float(stats_list[2 * i + idx]['num_loop_closures']) for i, rn in enumerate(run_names)}
        ).reindex(run_names)

    def make_combined_col(idx: int, col_name: str) -> TableData:
        # Only the successful count is ranked; the total is display-only.
        successful_table = make_highlighted_table(pd.DataFrame({col_name: successful_series(idx)}), "Method", fmt=int_fmt)
        total_table = make_highlighted_table(pd.DataFrame({col_name: total_series(idx)}), "Method", fmt=int_fmt, emphasis_rankings=False)
        return TableData.merge_TableData(successful_table, total_table)

    # Combined tables: one per LC side, columns match PDF table titles
    table_l = make_highlighted_table(pd.DataFrame({COL_SR_ALL: sr_series(0)}), "Method", fmt=percent_fmt) \
        .append_TableData(make_combined_col(0, COL_CNT_ALL), axis=1)
    table_r = make_highlighted_table(pd.DataFrame({COL_SR_INL: sr_series(1)}), "Method", fmt=percent_fmt) \
        .append_TableData(make_combined_col(1, COL_CNT_INL), axis=1)

    def _ate_suppressed(rn: str) -> bool:
        result = results[rn].get(col)
        lc_stats = result.lc_inlier_stats_by_mode.get(LCFilterMode.ONLY_INTER_LC) if result is not None else None
        return (lc_stats or {}).get('num_loop_closures', -1) == 0

    def _ate(rn: str) -> Optional[float]:
        result = results[rn].get(col)
        return result.merged_metrics.APE.translation_part.rmse if result is not None else None

    ate_raw = pd.Series(
        {rn: float('nan') if _ate_suppressed(rn) or _ate(rn) is None else _ate(rn)
         for rn in run_names}
    ).reindex(run_names)

    table_ate = make_highlighted_table(
        pd.DataFrame({COL_ATE: ate_raw}), "Method",
        color_fn=TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m), fmt=TableData.fmt_fixed(2), higher_is_better=False)

    # Scatter in center
    LoopClosureData.visualize_error_scatter(
        lc_data_list, labels_list, group_indices=group_indices,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        show_plots=False, ax=ax_center)

    # Pair name top-left in golden
    fig.text(0.01, 0.97, f"{pair[0]} / {pair[1]}",
             fontsize=40, fontweight='bold', va='top', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)

    # bbox=[x0, y0, width, height] in axes coordinates; height fraction limits row height.
    # Top table sits at the bottom of its axes; bottom table sits at the top of its axes
    # so they appear close together across the hspace gap.
    _bbox_lc_top  = [0, 0.0,  0.9, 0.45]
    _bbox_lc_bot  = [0, 0.50, 0.9, 0.45]
    _bbox_ate = [0, 0.30, 0.75, 0.30]   # ATE table: shorter axes, larger fraction needed

    # Section labels drawn just above their table's bbox top (y0 + height + small gap)
    ax_l.text(0.45, _bbox_lc_top[1] + _bbox_lc_top[3] + 0.02, "After Alignment",
              transform=ax_l.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')
    ax_r.text(0.45, _bbox_lc_bot[1] + _bbox_lc_bot[3] + 0.02, "After Kimera-RPGO",
              transform=ax_r.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')

    table_l.render_onto_ax(fig, ax_l, tbl_bbox=_bbox_lc_top, data_font_size=16)
    table_r.render_onto_ax(fig, ax_r, tbl_bbox=_bbox_lc_bot, data_font_size=16)
    table_ate.render_onto_ax(fig, ax_ate, tbl_bbox=_bbox_ate, font_size=16, data_font_size=20)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / f'lc_context_{col}.pdf'), bbox_inches='tight')
    plt.close(fig)


def _generate_lc_side_by_side_figure(
    pair: Tuple[str, str],
    col: str,
    lc_data_list: List[LoopClosureData],
    labels_list: List[str],
    group_indices: List[int],
    run_names: List[str],
    save_dir: Path,
) -> None:
    """Generate a side-by-side LC scatter slide for one robot pair.

    Produces a 22×12 inch figure with two equal scatter plots:
    - Left:  all loop closures plotted with X markers.
    - Right: inlier loop closures plotted with star markers.
    Both axes are forced square and share synchronized axis limits so errors
    are directly comparable. The pair name is shown in the top-left corner.

    Args:
        pair: Two robot names identifying this pair (e.g. ``("Husky1", "Drone1")``).
        col: Short pair label used in the filename (e.g. ``"H1D1"``).
        lc_data_list: Interleaved list of all-LC and inlier-LC LoopClosureData for
            each run (length ``2 * len(run_names)``), each with ``calculate_errors``
            and ``label_successful`` already called. Even indices are all-LC; odd
            are inlier-LC.
        labels_list: Display label for each entry in ``lc_data_list``.
        group_indices: Group index per entry, following the pattern
            ``[0, 0, 1, 1, ..., n-1, n-1]``.
        run_names: Ordered list of run identifiers.
        save_dir: Directory in which to save ``lc_side_by_side_<col>.pdf``.
    """
    expected_group_indices = [i for i in range(len(run_names)) for _ in range(2)]
    assert group_indices == expected_group_indices, (
        f"group_indices must be interleaved pairs [0,0,1,1,...], "
        f"got {group_indices}, expected {expected_group_indices}")

    lc_data_all    = lc_data_list[0::2]
    labels_all     = labels_list[0::2]
    lc_data_inlier = lc_data_list[1::2]
    labels_inlier  = [l.replace(" [Inliers]", "") for l in labels_list[1::2]]
    inlier_masks   = [np.ones(len(lc.results.translation_errors), dtype=bool) for lc in lc_data_inlier]

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.07, right=0.97, bottom=0.08, top=0.92, wspace=0.25)
    ax_all    = fig.add_subplot(gs[0])
    ax_inlier = fig.add_subplot(gs[1])
    ax_all.set_box_aspect(1)
    ax_inlier.set_box_aspect(1)

    LoopClosureData.visualize_error_scatter(
        lc_data_all, labels_all,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        show_plots=False, ax=ax_all)
    LoopClosureData.visualize_error_scatter(
        lc_data_inlier, labels_inlier,
        inlier_masks=inlier_masks,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        show_plots=False, ax=ax_inlier)

    # Synchronize axis limits so errors are directly comparable
    x_min = min(ax_all.get_xlim()[0], ax_inlier.get_xlim()[0])
    x_max = max(ax_all.get_xlim()[1], ax_inlier.get_xlim()[1])
    y_min = min(ax_all.get_ylim()[0], ax_inlier.get_ylim()[0])
    y_max = max(ax_all.get_ylim()[1], ax_inlier.get_ylim()[1])
    for ax in (ax_all, ax_inlier):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    ax_all.set_title("All Loop Closures", fontsize=16, fontweight='bold')
    ax_inlier.set_title("Inlier Loop Closures", fontsize=16, fontweight='bold')

    fig.text(0.01, 0.97, f"{pair[0]} / {pair[1]}",
             fontsize=40, fontweight='bold', va='top', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / f'lc_side_by_side_{col}.pdf'), bbox_inches='tight')
    plt.close(fig)


def _generate_traj_lc_comb_figure(
    col: str,
    run_names: List[str],
    traj_lc_dir: Path,
    save_dir: Path,
) -> None:
    """Combine the per-method traj_lc PDFs for one robot pair into a grid slide.

    Loads the existing ``traj_lc_<col>_<method>.pdf`` for each entry in
    ``run_names`` (no re-rendering) and places each page, as vector content,
    into one cell of a PowerPoint-widescreen-sized (13.333x7.5 in) PDF page.
    The grid shape is chosen to fit ``len(run_names)`` as close to square as
    possible (e.g. 4 methods -> 2x2, 5-6 methods -> 2x3). Each source page is
    scaled to fit its cell while preserving aspect ratio and centered within it.

    Args:
        col: Short pair label used in the filenames (e.g. ``"H1D1"``).
        run_names: Ordered list of run identifiers; index order maps
            left-to-right, top-to-bottom across the grid.
        traj_lc_dir: Directory containing the source ``traj_lc_<col>_<method>.pdf``
            files.
        save_dir: Directory in which to save ``traj_lc_comb_<col>.pdf``.
    """
    slide_width, slide_height = 960.0, 540.0  # 13.333x7.5 in @ 72 pt/in
    ncols = math.ceil(math.sqrt(len(run_names)))
    nrows = math.ceil(len(run_names) / ncols)
    cell_width, cell_height = slide_width / ncols, slide_height / nrows

    slide_doc = fitz.open()
    slide_page = slide_doc.new_page(width=slide_width, height=slide_height)

    for i, run_name in enumerate(run_names):
        src_doc = fitz.open(str(traj_lc_dir / f'traj_lc_{col}_{run_name}.pdf'))
        src_rect = src_doc[0].rect

        row, grid_col = divmod(i, ncols)
        scale = min(cell_width / src_rect.width, cell_height / src_rect.height)
        target_width, target_height = src_rect.width * scale, src_rect.height * scale
        x0 = grid_col * cell_width + (cell_width - target_width) / 2
        y0 = row * cell_height + (cell_height - target_height) / 2
        target_rect = fitz.Rect(x0, y0, x0 + target_width, y0 + target_height)

        slide_page.show_pdf_page(target_rect, src_doc, 0)
        src_doc.close()

    save_dir.mkdir(parents=True, exist_ok=True)
    slide_doc.save(str(save_dir / f'traj_lc_comb_{col}.pdf'))
    slide_doc.close()


def run_ROMAN_evaluation(roman_root: Path, dataset_prefix: str, dataset_name: str, run_names: List[str], all_robots: List[str],
                        critical_invocation_params: Dict[str, Any],
                        figures_base_dir: Path, load_gt_data_fn, viz_config: Dict,
                        ate_threshold_m: float, rot_threshold_deg: float = 10.0) -> None:
    """
    Generate all evaluation figures and tables for one dataset.

    For each robot pair across all run names:
      - Computes merged RMS ATE (pre- and post-optimize) in parallel.
      - For each ``LCFilterMode``, loads loop closure data under that filter,
        saves per-pair LC error scatter plots (lc/) and success-rate plots
        (lc_success_rate/).
      - Saves a context figure combining the LC scatter with per-run stats and
        ATE for each pair (lc_with_context/).

    Args:
        roman_root: Path to the roman repo checkout.
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``).
        dataset_name: Dataset identifier (e.g. ``"V2.3.AC"``).
        run_names: Ordered list of run/method identifiers to evaluate.
        all_robots: All robot names in the dataset; every pairwise combination is evaluated.
        critical_invocation_params: Other data-affecting args from the original run invocation.
        figures_base_dir: Directory under which ``figures/<dataset_prefix>/<dataset_name>/`` outputs are saved.
        load_gt_data_fn: Callable ``(dataset_name, robot_names) -> List[OdometryData]``,
            dataset-specific.
        viz_config: Dict forwarded to :func:`calculate_merged_ate` (see its docstring).
        ate_threshold_m: Red-highlight cutoff (m) for every translation-error table
            (ATE pre/post-optimize, individual ATE/RPE, RTE). Dataset-specific -- e.g.
            a smaller-area dataset like AirMuseum should use a smaller value than Hercules.
        rot_threshold_deg: Red-highlight cutoff (deg) for every rotation-error table
            (absolute and relative). Defaults to 10 degrees for all datasets.

    Outputs saved under ``figures/<dataset_prefix>/<dataset_name>/``:
      - ``metrics_table.pdf``  — pre/post-optimize RMS ATE, absolute/relative rotation error, and RTE summary tables
      - ``ate_split_table.pdf`` — per-robot RMS ATE/RPE summary tables, two columns per
        robot pair
      - ``timing_table.pdf``   — alignment/offline RPGO/total runtime summary tables
      - ``data_size_table.pdf``, ``data_size_table.tex`` — estimated communication data size (MB) summary table
      - ``mg_match_table.pdf`` — MG two-stage matcher stage-count summary table
      - ``traj/``              — per-pair estimated vs. GT trajectory plots

    Outputs saved under ``figures/<dataset_prefix>/<dataset_name>/<LCFilterMode.name>/``, once per LC filter mode:
      - ``lc_tables.pdf``      — LC success rate and count summary tables
      - ``lc_success_rate_table.tex``, ``lc_successful_total_table.tex`` — under
        ``LCFilterMode.ALL`` only, standalone LaTeX versions of the all-LC
        success rate and successful/total tables
      - ``lc/<pair>.pdf``      — per-pair LC error scatter plots
      - ``lc_success_rate/``   — per-pair LC success rate plots
      - ``lc_with_context/``   — per-pair composite slide figures
      - ``lc_side_by_side/``   — per-pair all-LC vs inlier-LC side-by-side scatter slides
      - ``traj_lc/``           — per-pair estimated trajectory with LC overlay
      - ``traj_lc_comb/``      — per-pair 2x2 combination of the per-method traj_lc slides
    """

    # Get all robot pairs to evaluation
    robot_pairs = list(itertools.combinations(all_robots, 2))

    # Load each run's SystemParams once, up front, and reuse it across every pair/table below
    system_params_by_run = {run_name: load_system_params_ROMAN(roman_root, dataset_prefix, dataset_name, run_name)
                            for run_name in run_names}

    # Define mapping between run name and display name
    run_display_names = {
        "ROMAN": "ROMAN (HERCULES replication)",
        "ROMAN_O": "ROMAN",
        "ROMAN_NM": "NM + ROMAN",
        "MG": "MeronomyGraph (Holonym Matching Only)",
        "MG_TS": "MeronomyGraph"
    }

    # Calculate RMS ATE
    tasks = [(roman_root, system_params_by_run[run_name], dataset_prefix, dataset_name, run_name, list(pair),
             critical_invocation_params, load_gt_data_fn, figures_base_dir, True, False, viz_config)
             for pair in robot_pairs
             for run_name in run_names]
    with Pool() as pool:
        pool_results = pool.starmap(calculate_merged_ate, tasks)

    # All computed results for this dataset, keyed by run then robot-pair column —
    # the single object threaded through every table/figure function below.
    results: Dict[str, Dict[str, ROMANResults]] = {run: {} for run in run_names}
    for (_, _, _, _, run_name, pair, *_), result in zip(tasks, pool_results):
        results[run_name][pair_label(*pair)] = result

    # Define sequence pair column names
    cols = [pair_label(*p) for p in robot_pairs]

    base_dir = Path(figures_base_dir) / dataset_prefix / dataset_name

    # Load per-pair runtime, data size, and MG match stats for every run
    for run_name in run_names:
        system_params = system_params_by_run[run_name]
        for pair in robot_pairs:
            col = pair_label(*pair)
            result = results[run_name][col]
            result.timing = load_timing_data_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                    list(pair), critical_invocation_params)
            result.data_size_mb = load_data_size_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                        list(pair), critical_invocation_params)
            result.mg_match = load_mg_match_stats_ROMAN(roman_root, system_params, dataset_prefix, dataset_name,
                                                         list(pair), critical_invocation_params)

    _save_timing_table(run_names, cols, run_display_names, results, base_dir / 'timing_table.pdf')
    _save_data_size_table(run_names, cols, run_display_names, results, base_dir / 'data_size_table.pdf')
    _save_mg_match_table(run_names, run_display_names, cols, results, base_dir / 'mg_match_table.pdf')

    # Generate the LC-dependent outputs once per LC filter mode, each under its own subfolder.
    # Process ONLY_INTER_LC first so its inlier-LC stats are available for the ATE
    # suppression logic in _generate_lc_context_figure across all modes.
    for lc_filter in sorted(LCFilterMode, key=lambda m: m != LCFilterMode.ONLY_INTER_LC):
        mode_dir = base_dir / lc_filter.name
        subdirs = {name: mode_dir / name for name in
                  ('lc', 'lc_success_rate', 'lc_with_context', 'lc_side_by_side', 'traj_lc', 'traj_lc_comb')}
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

        # For each pair...
        for pair in robot_pairs:
            # Load GT Data
            col = pair_label(*pair)
            gt_list = load_gt_data_fn(dataset_name, list(pair))
            gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

            # Calculate LC errors and visualize
            lc_data_list: List[LoopClosureData] = []
            labels_list: List[str] = []
            group_indices: List[int] = []
            for i, run_name in enumerate(run_names):
                merged_lc, merged_lc_inlier = load_LC_data_ROMAN(roman_root, system_params_by_run[run_name], dataset_prefix, dataset_name,
                                                                 list(pair), critical_invocation_params, lc_filter=lc_filter)
                for lc in (merged_lc, merged_lc_inlier):
                    lc.calculate_errors(gt_dict)
                    lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)
                lc_data_list.extend([merged_lc, merged_lc_inlier])
                labels_list.extend([run_name, run_name + " [Inliers]"])
                group_indices.extend([i, i])

            _, stats = LoopClosureData.visualize_error_scatter(
                lc_data_list, labels_list, group_indices=group_indices,
                max_rotation_frac=1.0, max_translation_frac=1.0,
                show_plots=False, save_path=str(subdirs['lc'] / f'lc_{col}.pdf'))

            fig_sr = LoopClosureData.visualize_success_rate(
                lc_data_list[::2], labels_list[::2], show_plots=False,
                max_translation_frac=0.01, max_rotation_frac=0.035, include_rate_plots=False)
            fig_sr.savefig(str(subdirs['lc_success_rate'] / f'lc_{col}_success_rate.pdf'))
            plt.close(fig_sr)

            for i, run_name in enumerate(run_names):
                results[run_name][col].lc_stats_by_mode[lc_filter] = stats[2 * i]
                results[run_name][col].lc_inlier_stats_by_mode[lc_filter] = stats[2 * i + 1]

            _generate_lc_context_figure(pair, col, lc_data_list, labels_list, group_indices,
                                        stats, results, run_names, subdirs['lc_with_context'], ate_threshold_m)
            _generate_lc_side_by_side_figure(pair, col, lc_data_list, labels_list, group_indices,
                                             run_names, subdirs['lc_side_by_side'])
            _generate_traj_lc_comb_figure(col, run_names, subdirs['traj_lc'], subdirs['traj_lc_comb'])

        _save_lc_tables(run_names, run_display_names, results, lc_filter, mode_dir / 'lc_tables.pdf')

    # ATE table is LC-independent, so it's saved once at the dataset root. Cell suppression
    # (no LC present) is based on inter-robot LC only, since only inter-robot closures actually
    # connect the pair's pose graph — intra-robot closures don't merge the two robots' trajectories.
    _save_ate_tables(run_names, cols, run_display_names, results, base_dir / 'metrics_table.pdf',
                     ate_threshold_m, rot_threshold_deg)

    # Per-robot RMS ATE/RPE split, also LC-independent and saved once at the dataset root.
    _save_ate_split_table(run_names, robot_pairs, run_display_names, results,
                          base_dir / 'ate_split_table.pdf', ate_threshold_m)

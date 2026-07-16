from enum import Enum
from evo.core.units import Unit
import getpass
import itertools
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import fitz
import pandas as pd
import re
from robotdataprocess import CoordinateFrame, LoopClosureData, OdometryData, PathData
from typing import Dict, List, Tuple

from utils.visualization import save_styled_tables, render_tables_onto_axes, HEADER_COLOR


def pair_label(name_a: str, name_b: str) -> str:
    def abbrev(n):
        m = re.match(r'([A-Za-z]+)(\d+)', n)
        return (m.group(1)[0].upper() + m.group(2)) if m else n
    return abbrev(name_a) + abbrev(name_b)


def print_metrics(metrics_dictionary: Dict) -> None:
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

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

def load_est_data_ROMAN(dataset_prefix: str, dataset_name: str, method: str, robot_names: List) -> List[OdometryData]:
    """
    Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

    Args:
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``).

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    user = getpass.getuser()
    run_name = "_".join(robot_names)
    return [
        OdometryData.from_csv(
            '/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + method +
            '/' + run_name + '/offline_rpgo/' + rn + '.csv',
            "map", 'robot' + str(i), CoordinateFrame.NONE, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        for i, rn in enumerate(robot_names)
    ]

def load_kimera_rpgo_first_stage_est_data_ROMAN(dataset_prefix: str, dataset_name: str, method: str, robot_names: List) -> List[OdometryData]:
    """
    Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

    Args:
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``).

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    user = getpass.getuser()
    run_name = "_".join(robot_names)
    result_dir = '/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + method + '/' + \
                  run_name + '/offline_rpgo/'
    names_override = {chr(97 + i): name for i, name in enumerate(robot_names)}
    return [
        OdometryData.from_g2o(result_dir + 'pre_optimize/result.g2o', result_dir + 'dense/odom_all.time.txt', rn,
            "map", 'robot' + str(i), CoordinateFrame.NONE, names_override)
        for i, rn in enumerate(robot_names)
    ]

def load_LC_data_ROMAN(dataset_prefix: str, dataset_name: str, run_name: str, robot_names: List,
                       lc_filter: LCFilterMode = LCFilterMode.ALL, names_override: Dict = None):
    """
    Load LC data for a ROMAN run.

    Args:
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``). The run folder is resolved as
            ``ROMAN_DEVEL/results/<dataset_prefix>_<dataset_name>_<run_name>/<robot0>_<robot1>``.
        names_override: If provided, maps g2o character keys ('a', 'b', ...) to robot names used in
            the returned LoopClosureData. Defaults to the original robot_names. Pass display-name
            overrides when LC names must match a visualize_2D nameList.

    Returns:
        (merged_lc, merged_lc_inlier)
    """
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + run_name + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    if lc_filter == LCFilterMode.ONLY_INTER_LC:
        pair_fn = itertools.combinations
    elif lc_filter == LCFilterMode.ONLY_INTRA_LC:
        pair_fn = lambda names, num: [(name, name) for name in names]
    else:
        pair_fn = itertools.combinations_with_replacement

    effective_override = names_override if names_override is not None else \
        {chr(97 + i): name for i, name in enumerate(robot_names)}

    # odom_and_lc.g2o already contains all robot pairs — load it once to avoid
    # tripling the count when iterating over combinations_with_replacement.
    merged_lc = LoopClosureData.from_g2o(
        run_folder / 'offline_rpgo' / 'dense' / 'odom_and_lc.g2o',
        run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
        names_override=effective_override)
    if lc_filter == LCFilterMode.ONLY_INTER_LC:
        merged_lc.prune_intra_robot_loop_closures()
    elif lc_filter == LCFilterMode.ONLY_INTRA_LC:
        merged_lc.prune_inter_robot_loop_closures()

    # Load the per-pair inlier g2o files (these are pair-specific)
    lc_inlier_data_list = []
    for name_a, name_b in pair_fn(robot_names, 2):
        letter_a = chr(97 + robot_names.index(name_a))
        letter_b = chr(97 + robot_names.index(name_b))
        if name_a == name_b:
            g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else:
            g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' / g2o_filename,
                                                  run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
                                                  names_override=effective_override)
        lc_inlier_data_list.append(lc_data_inlier)

    merged_lc_inlier = LoopClosureData.merge(lc_inlier_data_list)

    # Get information on duplicates and then prune them
    #merged_lc.print_duplicate_info(f"{dataset_name} {run_name} {robot_names} ALL")
    #merged_lc_inlier.print_duplicate_info(f"{dataset_name} {run_name} {robot_names} Kimera-RPGO Inlier")

    merged_lc.prune_duplicates()
    merged_lc_inlier.prune_duplicates()

    return merged_lc, merged_lc_inlier

def load_timing_data_ROMAN(dataset_prefix: str, dataset_name: str, method: str, robot_names: List) -> Dict[str, float]:
    """
    Load the runtime (s) breakdown for a ROMAN run on a robot pair.

    Reads the per-pair alignment runtimes (``align/runtime.txt``, one
    ``"key: value"`` line per intra-/inter-robot combination) and the offline
    RPGO optimization runtime (``offline_rpgo/runtime.txt``, a single value on
    its last non-empty line), kept separate so callers can report or combine
    them as needed.

    Returns:
        Dict with keys ``"align"`` (sum of the alignment runtimes) and
        ``"offline_rpgo"``, or ``None`` if either runtime file is missing or
        empty.
    """
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + method + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    align_runtime_path = run_folder / 'align' / 'runtime.txt'
    rpgo_runtime_path = run_folder / 'offline_rpgo' / 'runtime.txt'
    if not align_runtime_path.exists() or not rpgo_runtime_path.exists():
        return None

    align_lines = [line.strip() for line in align_runtime_path.read_text().splitlines() if line.strip()]
    rpgo_lines = [line.strip() for line in rpgo_runtime_path.read_text().splitlines() if line.strip()]
    if not align_lines or not rpgo_lines:
        return None

    align_total = sum(float(line.split(':')[-1]) for line in align_lines)
    rpgo_total = float(rpgo_lines[-1])

    return {"align": align_total, "offline_rpgo": rpgo_total}


def load_data_size_ROMAN(dataset_prefix: str, dataset_name: str, method: str, robot_names: List) -> float:
    """
    Load the estimated communication data size (decimal MB, 1 MB = 1,000,000 bytes)
    for a ROMAN run on a robot pair.

    Reads ``align/<robot0>_<robot1>/align.data_size.txt``, a single
    ``"Total submap data size (bytes): <value>"`` line.

    Returns:
        The data size in decimal MB (not MiB), or ``None`` if the file is missing.
    """
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + method + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    data_size_path = run_folder / 'align' / (robot_names[0] + '_' + robot_names[1]) / 'align.data_size.txt'
    if not data_size_path.exists():
        return None

    line = data_size_path.read_text().strip()
    data_size_bytes = float(line.split(':')[-1])
    return data_size_bytes / 1_000_000


def load_mg_match_stats_ROMAN(dataset_prefix: str, dataset_name: str, method: str, robot_names: List) -> Dict:
    """
    Count MG two-stage matcher calls by stage for a robot pair.

    Reads ``align/<robot_a>_<robot_b>/align.mg_match.txt`` for each of the (up
    to three) intra-/inter-robot combinations under this pair's align folder,
    and tallies how many calls reached stage 0, 1, or 2. Also collects, across
    all stage-1/2 calls, the values of ``n_stage1_matches`` and, across all
    stage-2 calls, ``n_stage2_child_clipper``,
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
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/' + dataset_prefix + '_' + dataset_name + '_' + method + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

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
        mg_match_path = run_folder / 'align' / f'{name_a}_{name_b}' / 'align.mg_match.txt'
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


def calculate_merged_ate(dataset_prefix: str, dataset_name: str, method: str, robot_names: List,
                         load_gt_data_fn,
                         figures_base_dir: Path = None, visualize: bool = False, do_individual_calcs: bool = False,
                         viz_config: Dict = None,
                         rpe_delta: float = 5.0, rpe_delta_unit: Unit = Unit.meters) -> Tuple:
    """
    Compute the merged RMS ATE for a robot pair after ROMAN offline RPGO.

    Concatenates the estimated and ground-truth trajectories for both robots
    (after aligning their time windows), then computes ATE on the combined
    trajectory. Optionally also computes the pre-optimize (first-stage) ATE
    and saves trajectory / LC overlay plots.

    Args:
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``), forwarded to :func:`load_est_data_ROMAN`,
            :func:`load_kimera_rpgo_first_stage_est_data_ROMAN`, and :func:`load_LC_data_ROMAN`.
        dataset_name: Dataset identifier (e.g. ``"V2.3.AC"``).
        method: Run name used to locate the ROMAN result folder (e.g. ``"ROMAN"``).
        robot_names: Two-element list of robot names (e.g. ``["Husky1", "Drone1"]``).
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
            defaults to identity). Required when ``visualize`` is True.
        rpe_delta: Step size between the pose pairs used for all RPE calculations in
            this function (first-stage, merged, and individual). Does not affect ATE.
            Defaults to ``5.0``.
        rpe_delta_unit: Unit of ``rpe_delta``. Defaults to ``Unit.meters``, matching
            the fixed-distance RPE convention used by GrAco.

    Returns:
        Tuple of ``(first_stage_ate, merged_ate, individual_ate, individual_rpe)``
        where ``first_stage_ate`` is the pre-optimize RMS ATE in metres (``None``
        if the pre-optimize file is unavailable), ``merged_ate`` is the final
        post-optimize RMS ATE in metres computed on the merged (both-robot)
        trajectory, ``individual_ate`` is a list of per-robot post-optimize RMS ATE
        values (``['APE']['translation_part']['rmse']``), and ``individual_rpe`` is
        a list of per-robot post-optimize RMS RPE values
        (``['RPE']['translation_part']['rmse']``) -- both in the same order as
        ``robot_names``, computed by separating the merged aligned trajectories
        back into per-robot trajectories (via :meth:`PathData.seperate_PathData`)
        and calling :meth:`PathData.calculate_traj_errors` on each robot's
        already-aligned pair.

    Raises:
        ValueError: If ``robot_names`` does not have exactly two entries.
    """
    if len(robot_names) != 2:
        raise ValueError(f"robot_names must have exactly two entries, got {len(robot_names)}: {robot_names}")
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]

    est_data_lst: List[OdometryData] = load_est_data_ROMAN(dataset_prefix, dataset_name, method, robot_names)
    gt_data_lst: List[OdometryData] = load_gt_data_fn(dataset_name, robot_names)
    est_data_robot0, est_data_robot1 = est_data_lst
    gt_data_robot0, gt_data_robot1 = gt_data_lst

    # Calculate individual RMS ATE
    if do_individual_calcs:
        # TODO: Need to make start and end times match before individual RMS ATE as well;
        # if we ever use those results in a paper.
        #print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False,
                                                                                rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        #print_metrics(metrics_dictionary)

        #print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False,
                                                                                rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        #print_metrics(metrics_dictionary)

    # Calculate first-stage (pre-optimize) ATE
    first_stage_ate = None
    try:
        first_stage_est_lst = load_kimera_rpgo_first_stage_est_data_ROMAN(dataset_prefix, dataset_name, method, robot_names)
        first_stage_est_lst, first_stage_gt_lst = PathData.make_start_and_end_times_match(first_stage_est_lst, gt_data_lst)
        first_stage_est: PathData = PathData.concatenate_PathData(first_stage_est_lst)
        first_stage_gt: PathData = PathData.concatenate_PathData(first_stage_gt_lst)
        #print("\n========== First Stage for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
        first_stage_metrics, _, _ = OdometryData.align_and_calculate_traj_errors(first_stage_gt, first_stage_est, max_diff=0.1, visualize=False,
                                                                                 rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        #print_metrics(first_stage_metrics)
        first_stage_ate = first_stage_metrics['APE']['translation_part']['rmse']
    except Exception as e:
        print(f"Warning: Could not compute first-stage ATE for {dataset_name} {method}: {e}")

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

    individual_metrics = [
        PathData.calculate_traj_errors(gt_align, est_align, rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
        for gt_align, est_align in zip(gt_data_align_list, est_data_align_list)
    ]
    individual_ate = [m['APE']['translation_part']['rmse'] for m in individual_metrics]
    individual_rpe = [m['RPE']['translation_part']['rmse'] for m in individual_metrics]

    if visualize:
        image_path = viz_config["image_path"]
        x_edge = viz_config["x_edge"]
        name_map: Dict = viz_config.get("name_map") or {rn: rn for rn in robot_names}
        robot_name_to_color: Dict = viz_config["robot_name_to_color"]
        image_extent_offsets = viz_config.get("background_image_extent_offsets")

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
                           save_path=str(traj_dir / f'traj_{pair_lbl}_{method}.pdf'))

        # Plot estimated trajectories with LC overlay (no background, no GT), once per LC filter mode
        names_override_display = {chr(97 + i): name_map[rn] for i, rn in enumerate([robot0_name, robot1_name])}
        gt_dict_display = {name_map[robot0_name]: gt_data_robot0, name_map[robot1_name]: gt_data_robot1}
        est_dataList =  [est_data_align_robot0,       est_data_align_robot1]
        est_isGTList =  [               False,                        False]
        est_nameList =  [name_map[robot0_name], name_map[robot1_name]]
        est_colorList = [robot_name_to_color[name] for name in est_nameList]
        for lc_filter in LCFilterMode:
            _, lc_data_inlier = load_LC_data_ROMAN(dataset_prefix, dataset_name, method, [robot0_name, robot1_name],
                                                   lc_filter=lc_filter, names_override=names_override_display)
            lc_errors_viz = lc_data_inlier.calculate_errors(gt_dict_display)

            traj_lc_dir = base_dir / lc_filter.name / 'traj_lc'
            traj_lc_dir.mkdir(parents=True, exist_ok=True)
            PathData.visualize_2D(est_dataList, est_isGTList, est_colorList, est_nameList, no_background=True, line_width=1.0, show_grid=True,
                               loop_closure_data=lc_data_inlier, lc_errors=lc_errors_viz, lc_line_width=2.0, lc_errors_vmax=2.0,
                               title=f"{method} LC overlaid on trajectory",
                               save_path=str(traj_lc_dir / f'traj_lc_{pair_lbl}_{method}.pdf'))

    return first_stage_ate, metrics_dictionary['APE']['translation_part']['rmse'], individual_ate, individual_rpe


def _save_ate_tables(run_names: List[str], cols: List[str],
                     run_display_names: Dict[str, str],
                     table_data: Dict[str, Dict[str, float]],
                     first_stage_table_data: Dict[str, Dict],
                     table_data_lc: Dict[str, Dict[str, Dict]],
                     table_data_lc_inlier: Dict[str, Dict[str, Dict]],
                     save_path: Path) -> None:
    """
    Build and save the ATE summary PDF tables.

    Produces two tables — pre-optimize (first-stage) and post-optimize merged
    RMS ATE — styled so that cells with no loop closures or ATE > 20 m are
    highlighted in red. The pre-optimize table is suppressed for pairs with
    zero total LC; the post-optimize table for pairs with zero inlier LC.

    Args:
        run_names: Ordered list of run identifiers (e.g. ``["ROMAN", "MG_TS"]``).
        cols: Ordered list of robot-pair column labels (e.g. ``["H1H2", "H1D1"]``).
        run_display_names: Maps each run identifier to its display name in the table.
        table_data: Post-optimize RMS ATE keyed by run then column.
        first_stage_table_data: Pre-optimize RMS ATE keyed by run then column
            (``None`` values are rendered as ``"---"``).
        table_data_lc: All-LC stats dicts keyed by run then column; used to
            suppress pre-optimize cells where total LC count is zero.
        table_data_lc_inlier: Inlier-LC stats dicts keyed by run then column;
            used to suppress post-optimize cells where inlier LC count is zero.
        save_path: Destination PDF path.
    """
    def make_df(data: dict, lc_filter: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
        rows = {}
        for run in run_names:
            row = {}
            for col in cols:
                val = data[run].get(col)
                if val is None or lc_filter[run].get(col, {}).get('num_loop_closures', -1) == 0:
                    row[col] = "---"
                else:
                    row[col] = f"{val:.3f}"
            rows[run_display_names.get(run, run)] = row
        return pd.DataFrame(rows).T

    def make_rank_df(data: dict) -> pd.DataFrame:
        # Negate so that lower ATE → higher rank value → sorted first by _col_rank_groups
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: -(data[run][col] if data[run].get(col) is not None else float('inf'))
                for col in cols}
             for run in run_names}
        ).T

    dfs = [
        ("Merged RMS ATE (m) — Pre-Optimize", make_df(first_stage_table_data, table_data_lc), [make_rank_df(first_stage_table_data)]),
        ("Merged RMS ATE (m)", make_df(table_data, table_data_lc_inlier), [make_rank_df(table_data)]),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.5,
                       cell_is_red=lambda s: s == "---" or float(s) > 20)


def _save_ate_split_table(run_names: List[str], robot_pairs: List[Tuple[str, str]],
                          run_display_names: Dict[str, str],
                          split_ate_table_data: Dict[str, Dict[str, List[float]]],
                          split_rpe_table_data: Dict[str, Dict[str, List[float]]],
                          save_path: Path) -> None:
    """
    Build and save the per-robot RMS ATE/RPE split summary PDF tables.

    For each robot pair, produces two adjacent columns (one per robot) holding
    that robot's individual post-optimize RMS ATE/RPE, computed by separating
    the merged aligned trajectory back into per-robot trajectories (see
    :func:`calculate_merged_ate`).

    Args:
        run_names: Ordered list of run identifiers.
        robot_pairs: Ordered list of two-robot-name tuples, matching the pair
            order used to build ``split_ate_table_data``/``split_rpe_table_data``.
        run_display_names: Maps each run identifier to its display name in the table.
        split_ate_table_data: Per-robot RMS ATE (m), keyed by run then robot-pair
            column label (e.g. ``"H1D1"``), each value a two-element list
            ``[robot0_ate, robot1_ate]`` (``None`` if unavailable).
        split_rpe_table_data: Per-robot RMS RPE (m), same shape as
            ``split_ate_table_data``.
        save_path: Destination PDF path.
    """
    sub_cols = [f"{pair_label(a, b)}\n{name}" for a, b in robot_pairs for name in (a, b)]

    def make_df(data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        rows = {}
        for run in run_names:
            row = {}
            for a, b in robot_pairs:
                vals = data[run].get(pair_label(a, b))
                for i, name in enumerate((a, b)):
                    row[f"{pair_label(a, b)}\n{name}"] = "---" if vals is None else f"{vals[i]:.6f}"
            rows[run_display_names.get(run, run)] = row
        return pd.DataFrame(rows).T[sub_cols]

    def make_rank_df(data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        rows = {}
        for run in run_names:
            row = {}
            for a, b in robot_pairs:
                vals = data[run].get(pair_label(a, b))
                for i, name in enumerate((a, b)):
                    row[f"{pair_label(a, b)}\n{name}"] = -(vals[i] if vals is not None else float('inf'))
            rows[run_display_names.get(run, run)] = row
        return pd.DataFrame(rows).T[sub_cols]

    dfs = [
        ("Individual RMS ATE (m)", make_df(split_ate_table_data), [make_rank_df(split_ate_table_data)]),
        ("Individual RMS RPE (m) - Δ5m", make_df(split_rpe_table_data), [make_rank_df(split_rpe_table_data)]),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.5,
                       cell_is_red=lambda s: s == "---" or float(s) > 20)


def _save_timing_table(run_names: List[str], cols: List[str],
                       run_display_names: Dict[str, str],
                       timing_table_data: Dict[str, Dict[str, Dict]],
                       save_path: Path) -> None:
    """
    Build and save the runtime summary PDF table.

    Produces one table per run and robot pair for each of the alignment
    runtime, the offline RPGO runtime, and their per-pair total.

    Args:
        run_names: Ordered list of run identifiers.
        cols: Ordered list of robot-pair column labels.
        run_display_names: Maps each run identifier to its display name in the table.
        timing_table_data: Runtime breakdown dicts (as returned by
            :func:`load_timing_data_ROMAN`) keyed by run then column
            (``None`` for pairs with unavailable runtime files).
        save_path: Destination PDF path.
    """
    def get_val(run: str, col: str, key: str):
        entry = timing_table_data[run].get(col)
        if key == "total":
            return None if entry is None else entry["align"] + entry["offline_rpgo"]
        return None if entry is None else entry[key]

    def make_df(key: str) -> pd.DataFrame:
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: ("---" if get_val(run, col, key) is None else f"{get_val(run, col, key):.1f}")
                for col in cols}
             for run in run_names}
        ).T

    def make_rank_df(key: str) -> pd.DataFrame:
        # Negate so that lower runtime → higher rank value → sorted first
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: -(get_val(run, col, key) if get_val(run, col, key) is not None else float('inf'))
                for col in cols}
             for run in run_names}
        ).T

    dfs = [
        ("Alignment Runtime (s)", make_df("align"), [make_rank_df("align")]),
        ("Offline RPGO Runtime (s)", make_df("offline_rpgo"), [make_rank_df("offline_rpgo")]),
        ("Total Runtime (s)", make_df("total"), [make_rank_df("total")]),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.5,
                       cell_is_red=lambda s: s == "---")


def _save_data_size_table(run_names: List[str], cols: List[str],
                          run_display_names: Dict[str, str],
                          data_size_table_data: Dict[str, Dict[str, float]],
                          save_path: Path) -> None:
    """
    Build and save the estimated communication data size summary PDF table.

    Args:
        run_names: Ordered list of run identifiers.
        cols: Ordered list of robot-pair column labels.
        run_display_names: Maps each run identifier to its display name in the table.
        data_size_table_data: Data size (MB) keyed by run then column
            (``None`` for pairs with unavailable data size files).
        save_path: Destination PDF path.
    """
    def make_df() -> pd.DataFrame:
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: ("---" if data_size_table_data[run].get(col) is None else f"{data_size_table_data[run][col]:.2f}")
                for col in cols}
             for run in run_names}
        ).T

    def make_rank_df() -> pd.DataFrame:
        # Negate so that lower data size → higher rank value → sorted first
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: -(data_size_table_data[run][col] if data_size_table_data[run].get(col) is not None else float('inf'))
                for col in cols}
             for run in run_names}
        ).T

    dfs = [("Estimated Communication Data Size (MB)", make_df(), [make_rank_df()])]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.5,
                       cell_is_red=lambda s: s == "---")


def _save_lc_tables(run_names: List[str], run_display_names: Dict[str, str],
                    table_data_lc: Dict[str, Dict[str, Dict]],
                    table_data_lc_inlier: Dict[str, Dict[str, Dict]],
                    save_path: Path) -> None:
    """
    Build and save the LC summary PDF tables.

    Produces four tables: success rate and successful/total counts for all LC
    and for inlier LC, across all run names and robot pairs.

    Args:
        run_names: Ordered list of run identifiers.
        run_display_names: Maps each run identifier to its display name in the table.
        table_data_lc: Per-run, per-pair stats dicts for all LC.
        table_data_lc_inlier: Per-run, per-pair stats dicts for inlier LC.
        save_path: Destination PDF path.
    """
    def make_df(data: dict, key: str, fmt) -> pd.DataFrame:
        return pd.DataFrame(
            {run_display_names.get(run, run): {col: fmt(data[run][col][key]) for col in data[run]}
             for run in run_names}
        ).T

    def make_rank_df(data: dict, key: str) -> pd.DataFrame:
        return pd.DataFrame(
            {run_display_names.get(run, run): {col: float(data[run][col][key]) for col in data[run]}
             for run in run_names}
        ).T

    def make_combined_df(data: dict) -> pd.DataFrame:
        def fmt(s): return f"{s['num_successful_loop_closures']}/{s['num_loop_closures']}"
        return pd.DataFrame(
            {run_display_names.get(run, run): {col: fmt(data[run][col]) for col in data[run]}
             for run in run_names}
        ).T

    dfs = [
        ("LC Success Rate %",
         make_df(table_data_lc, "success_rate", lambda x: f"{x:.1f}%"),
         [make_rank_df(table_data_lc, "success_rate")]),
        ("LC Successful / Total",
         make_combined_df(table_data_lc),
         [make_rank_df(table_data_lc, "num_successful_loop_closures"),
          make_rank_df(table_data_lc, "num_loop_closures")]),
        ("Inlier LC Success Rate %",
         make_df(table_data_lc_inlier, "success_rate", lambda x: f"{x:.1f}%"),
         [make_rank_df(table_data_lc_inlier, "success_rate")]),
        ("Inlier LC Successful / Total",
         make_combined_df(table_data_lc_inlier),
         [make_rank_df(table_data_lc_inlier, "num_successful_loop_closures"),
          make_rank_df(table_data_lc_inlier, "num_loop_closures")]),
    ]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.5)


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

    fig.suptitle(title, fontsize=14, fontweight='bold', color=HEADER_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _save_mg_match_table(run_names: List[str], run_display_names: Dict[str, str], cols: List[str],
                         table_data_mg_match: Dict[str, Dict[str, Dict]],
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
        table_data_mg_match: Stats dicts (as returned by
            :func:`load_mg_match_stats_ROMAN`) keyed by run then column
            (``None`` for pairs with no MG match files).
        save_path: Destination PDF path.
    """
    def make_counts_df() -> pd.DataFrame:
        return pd.DataFrame(
            {run_display_names.get(run, run): {
                col: ("---" if stats is None else f"{stats['stage_counts'][1]}-{stats['stage_counts'][2]}")
                for col, stats in table_data_mg_match[run].items()}
             for run in run_names}
        ).T

    def make_percent_df() -> pd.DataFrame:
        def fmt(stats):
            if stats is None:
                return "---"
            counts = stats['stage_counts']
            total = counts[0] + counts[1] + counts[2]
            if total == 0:
                return "---"
            return "-".join(f"{100 * counts[s] / total:.1f}%" for s in (1, 2))

        return pd.DataFrame(
            {run_display_names.get(run, run): {col: fmt(stats) for col, stats in table_data_mg_match[run].items()}
             for run in run_names}
        ).T

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(save_path)) as pp:
        fig, axes = plt.subplots(2, 1, figsize=(12, 2.4 * 2))
        for ax in axes:
            ax.axis('off')
        fig.tight_layout(pad=0.0, h_pad=0.5)
        render_tables_onto_axes(fig, [
            (axes[0], "MG Match Stage Counts (1-2)", make_counts_df(), None),
            (axes[1], "MG Match Stage Percentages (1-2)", make_percent_df(), None),
        ], cell_is_red=lambda s: s == "---")
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
                                 errors_list: List[Dict], labels_list: List[str],
                                 group_indices: List[int],
                                 stats_list: List[Dict],
                                 ate_table_data: Dict[str, Dict[str, float]],
                                 run_names: List[str], save_dir: Path) -> None:
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

    Table styling is applied via :func:`render_tables_onto_axes`.  Column names
    match the PDF table titles produced by :func:`_save_ate_tables` and
    :func:`_save_lc_tables`.

    Args:
        pair: Two robot names identifying this pair (e.g. ``("Husky1", "Drone1")``).
        col: Short pair label used as the table column header and in the filename
            (e.g. ``"H1D1"``).
        errors_list: Interleaved list of all-LC and inlier-LC error dicts for
            each run (length ``2 * len(run_names)``).
        labels_list: Display label for each entry in ``errors_list``.
        group_indices: Group index for each entry, pairing all-LC and inlier-LC
            entries within the same run.  Must follow the pattern
            ``[0, 0, 1, 1, ..., n-1, n-1]``.
        stats_list: Interleaved per-run LC stats dicts as returned by
            :meth:`LoopClosureData.visualize_error_scatter` (length
            ``2 * len(run_names)``).  Even indices are all-LC; odd are inlier-LC.
        ate_table_data: Post-optimize RMS ATE keyed by run name then pair column.
        run_names: Ordered list of run identifiers.
        save_dir: Directory in which to save ``lc_context_<col>.pdf``.
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

    # Center: scatter (forced square box regardless of figure proportions)
    ax_center = fig.add_subplot(gs[1])
    ax_center.set_box_aspect(1)

    # Right: ATE table centered by itself
    ax_ate = fig.add_subplot(gs[2])

    # Column name constants matching the PDF table titles
    COL_SR_ALL     = "LC Success \n Rate %"
    COL_CNT_ALL    = "LC Successful \n/ Total"
    COL_SR_INL     = "Inlier LC \nSuccess Rate %"
    COL_CNT_INL    = "Inlier LC \nSuccessful / Total"
    COL_ATE        = "RMS ATE (m)"

    def _count(s: Dict) -> str:
        return f"{s['num_successful_loop_closures']}/{s['num_loop_closures']}"

    def _sr(s: Dict) -> str:
        return f"{s['success_rate']:.1f}%"

    # Combined DataFrames: one per LC side, columns match PDF table titles
    df_l = pd.DataFrame({
        COL_SR_ALL:  {rn: _sr(stats_list[2 * i])    for i, rn in enumerate(run_names)},
        COL_CNT_ALL: {rn: _count(stats_list[2 * i]) for i, rn in enumerate(run_names)},
    }).reindex(run_names)
    df_r = pd.DataFrame({
        COL_SR_INL:  {rn: _sr(stats_list[2 * i + 1])    for i, rn in enumerate(run_names)},
        COL_CNT_INL: {rn: _count(stats_list[2 * i + 1]) for i, rn in enumerate(run_names)},
    }).reindex(run_names)
    ate_vals = {rn: ate_table_data[rn].get(col) for rn in run_names}
    df_ate = pd.DataFrame({
        COL_ATE: {rn: f"{v:.2f}" if v is not None else "---"
                  for rn, v in ate_vals.items()}
    }).reindex(run_names)

    # Rank DataFrames — one per combined table (whole-cell mode, independent per column)
    rank_l = pd.DataFrame({
        COL_SR_ALL:  {rn: stats_list[2 * i]['success_rate']                          for i, rn in enumerate(run_names)},
        COL_CNT_ALL: {rn: float(stats_list[2 * i]['num_successful_loop_closures'])    for i, rn in enumerate(run_names)},
    }).reindex(run_names)
    rank_r = pd.DataFrame({
        COL_SR_INL:  {rn: stats_list[2 * i + 1]['success_rate']                          for i, rn in enumerate(run_names)},
        COL_CNT_INL: {rn: float(stats_list[2 * i + 1]['num_successful_loop_closures'])    for i, rn in enumerate(run_names)},
    }).reindex(run_names)
    rank_ate = pd.DataFrame({
        COL_ATE: {rn: -(v if v is not None else float('inf')) for rn, v in ate_vals.items()}
    }).reindex(run_names)

    # Scatter in center
    LoopClosureData.visualize_error_scatter(
        errors_list, labels_list, group_indices=group_indices,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        trans_err_in_target=1.0, rot_err_in_target=5.0,
        show_plots=False, ax=ax_center)

    # Pair name top-left in golden
    fig.text(0.01, 0.97, f"{pair[0]} / {pair[1]}",
             fontsize=40, fontweight='bold', va='top', color=HEADER_COLOR)

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

    render_tables_onto_axes(fig, [
        (ax_l,   "Method", df_l,   [rank_l],   None,                                  _bbox_lc_top, None,             16),
        (ax_r,   "Method", df_r,   [rank_r],   None,                                  _bbox_lc_bot, None,             16),
        (ax_ate, "Method", df_ate, [rank_ate], lambda s: s == "---" or float(s) > 20, _bbox_ate, 16, 20),
    ])

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_dir / f'lc_context_{col}.pdf'), bbox_inches='tight')
    plt.close(fig)


def _generate_lc_side_by_side_figure(
    pair: Tuple[str, str],
    col: str,
    errors_list: List[Dict],
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
        errors_list: Interleaved list of all-LC and inlier-LC error dicts for each
            run (length ``2 * len(run_names)``). Even indices are all-LC; odd are
            inlier-LC.
        labels_list: Display label for each entry in ``errors_list``.
        group_indices: Group index per entry, following the pattern
            ``[0, 0, 1, 1, ..., n-1, n-1]``.
        run_names: Ordered list of run identifiers.
        save_dir: Directory in which to save ``lc_side_by_side_<col>.pdf``.
    """
    expected_group_indices = [i for i in range(len(run_names)) for _ in range(2)]
    assert group_indices == expected_group_indices, (
        f"group_indices must be interleaved pairs [0,0,1,1,...], "
        f"got {group_indices}, expected {expected_group_indices}")

    errors_all    = errors_list[0::2]
    labels_all    = labels_list[0::2]
    errors_inlier = errors_list[1::2]
    labels_inlier = [l.replace(" [Inliers]", "") for l in labels_list[1::2]]
    inlier_masks  = [np.ones(len(e["translation_errors"]), dtype=bool) for e in errors_inlier]

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.07, right=0.97, bottom=0.08, top=0.92, wspace=0.25)
    ax_all    = fig.add_subplot(gs[0])
    ax_inlier = fig.add_subplot(gs[1])
    ax_all.set_box_aspect(1)
    ax_inlier.set_box_aspect(1)

    LoopClosureData.visualize_error_scatter(
        errors_all, labels_all,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        trans_err_in_target=1.0, rot_err_in_target=5.0,
        show_plots=False, ax=ax_all)
    LoopClosureData.visualize_error_scatter(
        errors_inlier, labels_inlier,
        inlier_masks=inlier_masks,
        max_rotation_frac=1.0, max_translation_frac=1.0,
        trans_err_in_target=1.0, rot_err_in_target=5.0,
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
             fontsize=40, fontweight='bold', va='top', color=HEADER_COLOR)

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


def run_ROMAN_evaluation(dataset_prefix: str, dataset_name: str, run_names: List[str], all_robots: List[str],
                        figures_base_dir: Path, load_gt_data_fn, viz_config: Dict) -> None:
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
        dataset_prefix: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
            ``"GrAco"``).
        dataset_name: Dataset identifier (e.g. ``"V2.3.AC"``).
        run_names: Ordered list of run/method identifiers to evaluate.
        all_robots: All robot names in the dataset; every pairwise combination is evaluated.
        figures_base_dir: Directory under which ``figures/<dataset_prefix>/<dataset_name>/`` outputs are saved.
        load_gt_data_fn: Callable ``(dataset_name, robot_names) -> List[OdometryData]``,
            dataset-specific.
        viz_config: Dict forwarded to :func:`calculate_merged_ate` (see its docstring).

    Outputs saved under ``figures/<dataset_prefix>/<dataset_name>/``:
      - ``ate_table.pdf``      — pre/post-optimize RMS ATE summary tables
      - ``ate_split_table.pdf`` — per-robot RMS ATE/RPE summary tables, two columns per
        robot pair
      - ``timing_table.pdf``   — alignment/offline RPGO/total runtime summary tables
      - ``data_size_table.pdf`` — estimated communication data size (MB) summary table
      - ``mg_match_table.pdf`` — MG two-stage matcher stage-count summary table
      - ``traj/``              — per-pair estimated vs. GT trajectory plots

    Outputs saved under ``figures/<dataset_prefix>/<dataset_name>/<LCFilterMode.name>/``, once per LC filter mode:
      - ``lc_tables.pdf``      — LC success rate and count summary tables
      - ``lc/<pair>.pdf``      — per-pair LC error scatter plots
      - ``lc_success_rate/``   — per-pair LC success rate plots
      - ``lc_with_context/``   — per-pair composite slide figures
      - ``lc_side_by_side/``   — per-pair all-LC vs inlier-LC side-by-side scatter slides
      - ``traj_lc/``           — per-pair estimated trajectory with LC overlay
      - ``traj_lc_comb/``      — per-pair 2x2 combination of the per-method traj_lc slides
    """

    # Get all robot pairs to evaluation
    robot_pairs = list(itertools.combinations(all_robots, 2))

    # Define mapping between run name and display name
    run_display_names = {
        "ROMAN": "ROMAN",
        "ROMAN_Deduplication": "ROMAN w/o duplicate LC",
        "ROMAN_NM":   "NM + ROMAN",
        "ROMAN_NM_POA_Triplet": "NM + ROMAN + Triplet POA",
        "MG": "MeronomyGraph",
        "MG_TS": "NM + MG",
        "MG_TS_noGM": "NM + MG (no Global Map)",
        "MG_TS_Duplication": "NM + MG (Above but with Dup. LC)",
        "MG_TS_<Old_Version>": "NM + MG (Two Stage - Reworked 4)",
        "MG_TS_2-4":  "NM + MG (Two Stage - 2/4 req)",
        "MG_TS_3-4":  "NM + MG (Two Stage - 3/4 req)",
        "MG_SS_3":    "NM + MG (Single Stage - 3 req)",
        "MG_SS_3_POA":"NM + MG (SS3) + POA"
    }

    # Calculate RMS ATE
    tasks = [(dataset_prefix, dataset_name, run_name, list(pair), load_gt_data_fn,
             figures_base_dir, True, False, viz_config)
             for pair in robot_pairs
             for run_name in run_names]
    with Pool() as pool:
        results = pool.starmap(calculate_merged_ate, tasks)

    # Get RMS ATE and save in tables for  visualization
    table_data: Dict[str, Dict[str, float]] = {run: {} for run in run_names}
    first_stage_table_data: Dict[str, Dict] = {run: {} for run in run_names}
    split_ate_table_data: Dict[str, Dict[str, List[float]]] = {run: {} for run in run_names}
    split_rpe_table_data: Dict[str, Dict[str, List[float]]] = {run: {} for run in run_names}
    for (_, _, run_name, pair, *_), (first_stage_ate, ate, individual_ate, individual_rpe) in zip(tasks, results):
        col = pair_label(*pair)
        table_data[run_name][col] = ate
        first_stage_table_data[run_name][col] = first_stage_ate
        split_ate_table_data[run_name][col] = individual_ate
        split_rpe_table_data[run_name][col] = individual_rpe

    # Define sequence pair column names
    cols = [pair_label(*p) for p in robot_pairs]

    base_dir = Path(figures_base_dir) / dataset_prefix / dataset_name

    # Load runtime data and save the timing table
    timing_table_data: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}
    for run_name in run_names:
        for pair in robot_pairs:
            col = pair_label(*pair)
            timing_table_data[run_name][col] = load_timing_data_ROMAN(dataset_prefix, dataset_name, run_name, list(pair))
    _save_timing_table(run_names, cols, run_display_names, timing_table_data, base_dir / 'timing_table.pdf')

    # Load estimated communication data size and save the summary table
    data_size_table_data: Dict[str, Dict[str, float]] = {run: {} for run in run_names}
    for run_name in run_names:
        for pair in robot_pairs:
            col = pair_label(*pair)
            data_size_table_data[run_name][col] = load_data_size_ROMAN(dataset_prefix, dataset_name, run_name, list(pair))
    _save_data_size_table(run_names, cols, run_display_names, data_size_table_data, base_dir / 'data_size_table.pdf')

    # Load MG two-stage matcher stage counts and save the summary table
    table_data_mg_match: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}
    for run_name in run_names:
        for pair in robot_pairs:
            col = pair_label(*pair)
            table_data_mg_match[run_name][col] = load_mg_match_stats_ROMAN(dataset_prefix, dataset_name, run_name, list(pair))
    _save_mg_match_table(run_names, run_display_names, cols, table_data_mg_match, base_dir / 'mg_match_table.pdf')

    # Generate the LC-dependent outputs once per LC filter mode, each under its own subfolder
    table_data_lc_by_mode: Dict[LCFilterMode, Dict[str, Dict[str, Dict]]] = {}
    table_data_lc_inlier_by_mode: Dict[LCFilterMode, Dict[str, Dict[str, Dict]]] = {}

    for lc_filter in LCFilterMode:
        mode_dir = base_dir / lc_filter.name
        lc_dir = mode_dir / 'lc'
        lc_sr_dir = mode_dir / 'lc_success_rate'
        lc_with_context_dir = mode_dir / 'lc_with_context'
        lc_side_by_side_dir = mode_dir / 'lc_side_by_side'
        traj_lc_dir = mode_dir / 'traj_lc'
        traj_lc_comb_dir = mode_dir / 'traj_lc_comb'
        lc_dir.mkdir(parents=True, exist_ok=True)
        lc_sr_dir.mkdir(parents=True, exist_ok=True)
        lc_with_context_dir.mkdir(parents=True, exist_ok=True)
        lc_side_by_side_dir.mkdir(parents=True, exist_ok=True)
        traj_lc_comb_dir.mkdir(parents=True, exist_ok=True)

        # Create tables to hold LC results
        table_data_lc: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}
        table_data_lc_inlier: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}

        # For each pair...
        for pair in robot_pairs:
            # Load GT Data
            col = pair_label(*pair)
            gt_list = load_gt_data_fn(dataset_name, list(pair))
            gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

            # Calculate LC errors and visualize
            errors_list: List[Dict] = []
            labels_list: List[str] = []
            group_indices: List[int] = []
            for i, run_name in enumerate(run_names):
                merged_lc, merged_lc_inlier = load_LC_data_ROMAN(dataset_prefix, dataset_name, run_name, list(pair), lc_filter=lc_filter)
                errors_list.extend([merged_lc.calculate_errors(gt_dict), merged_lc_inlier.calculate_errors(gt_dict)])
                labels_list.extend([run_name, run_name + " [Inliers]"])
                group_indices.extend([i, i])

            _, stats = LoopClosureData.visualize_error_scatter(
                errors_list, labels_list, group_indices=group_indices,
                max_rotation_frac=1.0, max_translation_frac=1.0,
                trans_err_in_target=1.0, rot_err_in_target=5.0,
                show_plots=False, save_path=str(lc_dir / f'lc_{col}.pdf'))

            fig_sr = LoopClosureData.visualize_success_rate(
                errors_list[::2], labels_list[::2], show_plots=False,
                max_translation_frac=0.01, max_rotation_frac=0.035, include_rate_plots=False)
            fig_sr.savefig(str(lc_sr_dir / f'lc_{col}_success_rate.pdf'))
            plt.close(fig_sr)

            for i, run_name in enumerate(run_names):
                table_data_lc[run_name][col] = stats[2 * i]
                table_data_lc_inlier[run_name][col] = stats[2 * i + 1]

            _generate_lc_context_figure(pair, col, errors_list, labels_list, group_indices,
                                        stats, table_data, run_names, lc_with_context_dir)
            _generate_lc_side_by_side_figure(pair, col, errors_list, labels_list, group_indices,
                                             run_names, lc_side_by_side_dir)
            _generate_traj_lc_comb_figure(col, run_names, traj_lc_dir, traj_lc_comb_dir)

        _save_lc_tables(run_names, run_display_names, table_data_lc, table_data_lc_inlier,
                        mode_dir / 'lc_tables.pdf')

        table_data_lc_by_mode[lc_filter] = table_data_lc
        table_data_lc_inlier_by_mode[lc_filter] = table_data_lc_inlier

    # ATE table is LC-independent, so it's saved once at the dataset root. Cell suppression
    # (no LC present) is based on inter-robot LC only, since only inter-robot closures actually
    # connect the pair's pose graph — intra-robot closures don't merge the two robots' trajectories.
    _save_ate_tables(run_names, cols, run_display_names, table_data, first_stage_table_data,
                     table_data_lc_by_mode[LCFilterMode.ONLY_INTER_LC], table_data_lc_inlier_by_mode[LCFilterMode.ONLY_INTER_LC],
                     base_dir / 'ate_table.pdf')

    # Per-robot RMS ATE/RPE split, also LC-independent and saved once at the dataset root.
    _save_ate_split_table(run_names, robot_pairs, run_display_names, split_ate_table_data, split_rpe_table_data,
                          base_dir / 'ate_split_table.pdf')

import getpass
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import sys
from pathlib import Path
import pandas as pd
from robotdataprocess import LoopClosureData, OdometryData, PathData
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from utils.visualization import save_styled_tables, render_tables_onto_axes, HEADER_COLOR
from utils_ROMAN import _pair_label, load_gt_data_ROMAN, load_est_data_ROMAN, load_kimera_rpgo_first_stage_est_data_ROMAN

def _print_metrics(metrics_dictionary: Dict) -> None:
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

def load_LC_data_ROMAN(dataset_name: str, run_name: str, robot_names: List, only_inter_lc: bool = False,
                       names_override: Dict = None):
    """
    Load LC data for a ROMAN run.

    Args:
        names_override: If provided, maps g2o character keys ('a', 'b', ...) to robot names used in
            the returned LoopClosureData. Defaults to the original robot_names. Pass display-name
            overrides when LC names must match a visualize_2D nameList.

    Returns:
        (merged_lc, merged_lc_inlier)
    """
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + run_name + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    pair_fn = itertools.combinations if only_inter_lc else itertools.combinations_with_replacement

    effective_override = names_override if names_override is not None else \
        {chr(97 + i): name for i, name in enumerate(robot_names)}

    # odom_and_lc.g2o already contains all robot pairs — load it once to avoid
    # tripling the count when iterating over combinations_with_replacement.
    merged_lc = LoopClosureData.from_g2o(
        run_folder / 'offline_rpgo' / 'dense' / 'odom_and_lc.g2o',
        run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
        names_override=effective_override)
    if only_inter_lc:
        merged_lc.prune_intra_robot_loop_closures()

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
    return merged_lc, merged_lc_inlier

def calculate_merged_ate(dataset_name: str, method: str, robot_names: List, visualize: bool = False,
                         do_individual_calcs: bool = False) -> Tuple:
    """
    Compute the merged RMS ATE for a robot pair after ROMAN offline RPGO.

    Concatenates the estimated and ground-truth trajectories for both robots
    (after aligning their time windows), then computes ATE on the combined
    trajectory. Optionally also computes the pre-optimize (first-stage) ATE
    and saves trajectory / LC overlay plots.

    Args:
        dataset_name: Hercules dataset identifier (e.g. ``"V2.3.AC"``).
        method: Run name used to locate the ROMAN result folder (e.g. ``"ROMAN"``).
        robot_names: Two-element list of robot names (e.g. ``["Husky1", "Drone1"]``).
        visualize: If True, generate and save 2D trajectory and LC overlay PDFs.
        do_individual_calcs: If True, also print per-robot ATE before the merged calc.

    Returns:
        Tuple of ``(first_stage_ate, merged_ate)`` where ``first_stage_ate`` is the
        pre-optimize RMS ATE in metres (``None`` if the pre-optimize file is
        unavailable) and ``merged_ate`` is the final post-optimize RMS ATE in metres.
    """
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]

    est_data_lst: List[OdometryData] = load_est_data_ROMAN(dataset_name, method, robot_names)
    gt_data_lst: List[OdometryData] = load_gt_data_ROMAN(dataset_name, robot_names)
    est_data_robot0, est_data_robot1 = est_data_lst
    gt_data_robot0, gt_data_robot1 = gt_data_lst

    # Calculate individual RMS ATE
    if do_individual_calcs:
        # TODO: Need to make start and end times match before individual RMS ATE as well;
        # if we ever use those results in a paper.

        print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False)
        _print_metrics(metrics_dictionary)

        print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False)
        _print_metrics(metrics_dictionary)

    # Calculate first-stage (pre-optimize) ATE
    first_stage_ate = None
    try:
        first_stage_est_lst = load_kimera_rpgo_first_stage_est_data_ROMAN(dataset_name, method, robot_names)
        first_stage_est_lst, first_stage_gt_lst = PathData.make_start_and_end_times_match(first_stage_est_lst, gt_data_lst)
        first_stage_est: PathData = PathData.concatenate_PathData(first_stage_est_lst)
        first_stage_gt: PathData = PathData.concatenate_PathData(first_stage_gt_lst)
        print("\n========== First Stage for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
        first_stage_metrics, _, _ = OdometryData.align_and_calculate_traj_errors(first_stage_gt, first_stage_est, max_diff=0.1, visualize=False)
        _print_metrics(first_stage_metrics)
        first_stage_ate = first_stage_metrics['APE']['translation_part']['rmse']
    except Exception as e:
        print(f"Warning: Could not compute first-stage ATE for {dataset_name} {method}: {e}")

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    _print_metrics(metrics_dictionary)

    if visualize:
        # Seperate the aligned trajectories into their single-robot forms
        gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
        gt_data_align_robot0 = gt_data_align_list[0]
        gt_data_align_robot1 = gt_data_align_list[1]

        est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
        est_data_align_robot0 = est_data_align_list[0]
        est_data_align_robot1 = est_data_align_list[1]

        # Get environment image path
        user = getpass.getuser()
        image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/environment.png'
        if dataset_name in "V2.3.AP":  x_edge = 350
        elif dataset_name in "V2.4.C": x_edge = 300
        elif dataset_name in "V2.3.AC": x_edge = 500
        elif dataset_name in "V2.4.F": x_edge = 150
        else:
            raise RuntimeError(f"x_edge not defined for {dataset_name}.")

        # Define the mapping from robot name to color and robot_name to new name
        name_map: Dict = {
            "Husky1": "UGV1",
            "Husky2": "UGV2",
            "Drone1": "UAV1",
            "Drone2": "UAV2"
        }
        robot_name_to_color: Dict = {
            "UGV1": "#1EE15F",
            "UGV2": "#E11E28",
            "UAV1": "#F0F02A",
            "UAV2": "#1B0ED5",
        }

        # Load LC data with display names so names match nameList
        names_override_display = {chr(97 + i): name_map[rn] for i, rn in enumerate([robot0_name, robot1_name])}
        _, lc_data_inlier = load_LC_data_ROMAN(dataset_name, method, [robot0_name, robot1_name],
                                               only_inter_lc=True, names_override=names_override_display)
        gt_dict_display = {name_map[robot0_name]: gt_data_robot0, name_map[robot1_name]: gt_data_robot1}
        lc_errors_viz = lc_data_inlier.calculate_errors(gt_dict_display)

        pair_label = _pair_label(robot0_name, robot1_name)
        base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures') / dataset_name
        traj_dir = base_dir / 'traj'
        traj_lc_dir = base_dir / 'traj_lc'
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_lc_dir.mkdir(parents=True, exist_ok=True)

        # Plot the results in 2D (Configuration for Figure 10)
        dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
        isGTList =  [                False,                 True,                  False,                  True]
        nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True,
                           background_image_path=image_path, background_image_x_edge=x_edge,
                           save_path=str(traj_dir / f'traj_{pair_label}_{method}.pdf'))

        # Plot estimated trajectories with LC overlay (no background, no GT)
        est_dataList =  [est_data_align_robot0,       est_data_align_robot1]
        est_isGTList =  [               False,                        False]
        est_nameList =  [name_map[robot0_name], name_map[robot1_name]]
        est_colorList = [robot_name_to_color[name] for name in est_nameList]
        PathData.visualize_2D(est_dataList, est_isGTList, est_colorList, est_nameList, no_background=True, line_width=1.0, show_grid=True,
                           loop_closure_data=lc_data_inlier, lc_errors=lc_errors_viz, lc_line_width=2.0, lc_errors_vmax=2.0,
                           title=f"{method} LC overlaid on trajectory",
                           save_path=str(traj_lc_dir / f'traj_lc_{pair_label}_{method}.pdf'))

        # Configuration for Figure 2
        # PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=4.0, show_grid=False,
        #                     background_image_path=image_path, background_image_x_edge=x_edge, legend=False, no_border=True,
        #                     save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

    return first_stage_ate, metrics_dictionary['APE']['translation_part']['rmse']


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
                    row[col] = f"{val:.2f}"
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
    save_styled_tables(dfs, str(save_path), row_height=2.4, h_pad=0.05,
                       cell_is_red=lambda s: s == "---" or float(s) > 20)


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


def main():
    """
    Generate all evaluation figures and tables for the HERCULES dataset.

    For each robot pair across all run names:
      - Computes merged RMS ATE (pre- and post-optimize) in parallel.
      - Loads loop closure data, saves per-pair LC error scatter plots (lc/)
        and success-rate plots (lc_success_rate/).
      - Saves a context figure combining the LC scatter with per-run stats and
        ATE for each pair (lc_with_context/).

    Outputs saved under ``figures/<dataset_name>/``:
      - ``ate_table.pdf``      — pre/post-optimize RMS ATE summary tables
      - ``lc_tables.pdf``      — LC success rate and count summary tables
      - ``lc/<pair>.pdf``      — per-pair LC error scatter plots
      - ``lc_success_rate/``   — per-pair LC success rate plots
      - ``lc_with_context/``   — per-pair composite slide figures
      - ``lc_side_by_side/``   — per-pair all-LC vs inlier-LC side-by-side scatter slides
    """
    
    # Get all robot pairs to evaluation
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    robot_pairs = list(itertools.combinations(all_robots, 2))

    # Define the dataset and methods to evaluate
    run_names = ["ROMAN", "ROMAN_NM", "MG_TS_noGM", "MG_TS"]
    dataset_name = "V2.3.AC"

    # Calculate RMS ATE
    tasks = [(dataset_name, run_name, list(pair), True)
             for pair in robot_pairs
             for run_name in run_names]
    with Pool() as pool:
        results = pool.starmap(calculate_merged_ate, tasks)

    # Get RMS ATE and save in tables for  visualization
    table_data: Dict[str, Dict[str, float]] = {run: {} for run in run_names}
    first_stage_table_data: Dict[str, Dict] = {run: {} for run in run_names}
    for (_, run_name, pair, *_), (first_stage_ate, ate) in zip(tasks, results):
        col = _pair_label(*pair)
        table_data[run_name][col] = ate
        first_stage_table_data[run_name][col] = first_stage_ate

    # Define sequence pair column names
    cols = [_pair_label(*p) for p in robot_pairs]

    # Define parameter name to display name mapping
    RUN_DISPLAY_NAMES = {
        "ROMAN": "ROMAN",
        "ROMAN_Deduplication": "ROMAN w/o duplicate LC",
        "ROMAN_NM":   "NM + ROMAN",
        "ROMAN_NM_POA_Triplet": "NM + ROMAN + Triplet POA",
        "MG_TS": "NM + MG",
        "MG_TS_noGM": "NM + MG (no Global Map)",
        "MG_TS_Duplication": "NM + MG (Above but with Dup. LC)",
        "MG_TS_<Old_Version>": "NM + MG (Two Stage - Reworked 4)",
        "MG_TS_2-4":  "NM + MG (Two Stage - 2/4 req)",
        "MG_TS_3-4":  "NM + MG (Two Stage - 3/4 req)",
        "MG_SS_3":    "NM + MG (Single Stage - 3 req)",
        "MG_SS_3_POA":"NM + MG (SS3) + POA"
    }

    # Create all necessary directories
    base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures') / dataset_name
    lc_dir = base_dir / 'lc'
    lc_sr_dir = base_dir / 'lc_success_rate'
    lc_with_context_dir = base_dir / 'lc_with_context'
    lc_side_by_side_dir = base_dir / 'lc_side_by_side'
    lc_dir.mkdir(parents=True, exist_ok=True)
    lc_sr_dir.mkdir(parents=True, exist_ok=True)
    lc_with_context_dir.mkdir(parents=True, exist_ok=True)
    lc_side_by_side_dir.mkdir(parents=True, exist_ok=True)

    # Create tables to hold LC results
    table_data_lc: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}
    table_data_lc_inlier: Dict[str, Dict[str, Dict]] = {run: {} for run in run_names}

    # For each pair...
    for pair in robot_pairs:
        # Load GT Data
        col = _pair_label(*pair)
        gt_list = load_gt_data_ROMAN(dataset_name, list(pair))
        gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

        # Calculate LC errors and visualize
        errors_list: List[Dict] = []
        labels_list: List[str] = []
        group_indices: List[int] = []
        for i, run_name in enumerate(run_names):
            merged_lc, merged_lc_inlier = load_LC_data_ROMAN(dataset_name, run_name, list(pair), only_inter_lc=True)
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

    _save_ate_tables(run_names, cols, RUN_DISPLAY_NAMES, table_data, first_stage_table_data,
                     table_data_lc, table_data_lc_inlier, base_dir / 'ate_table.pdf')
    _save_lc_tables(run_names, RUN_DISPLAY_NAMES, table_data_lc, table_data_lc_inlier,
                    base_dir / 'lc_tables.pdf')


if __name__ == "__main__":
    main()

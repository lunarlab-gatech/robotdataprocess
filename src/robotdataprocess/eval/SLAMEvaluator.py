from evo.core.units import Unit
import fitz
import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import re
from robotdataprocess import LoopClosureData, LoopClosureFilterMode, OdometryData, PathData, TableData
from robotdataprocess.data_types.SLAMData import SLAMData
from robotdataprocess.eval.RobotGroup import RobotGroup
from robotdataprocess.eval.SLAMEvaluatorResult import SLAMResult
import seaborn as sns
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class SLAMEvaluator:
    """
    Evaluation of SLAM runs; specifically built for working with the MeronomyGraph repo. Operates
    on :class:`SLAMData` instances, which own the loading of the data being evaluated.
    """

    # =========================================================================
    # ============================= Table Helpers =============================
    # =========================================================================
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def group_label(names) -> str:
        """
        Build a short column label for an arbitrary-size group of robots (a
        singleton for self-alignment, a pair, or a larger group), joining each
        robot's abbreviation with '-' (e.g. ``("Husky1", "Drone1") -> "H1-D1"``).
        """
        def initials(core):
            # Word initials (split on non-alphanumeric separators, e.g. "_") disambiguate
            # multi-word names sharing a leading letter (e.g. "acl_jackal" -> "AJ" vs "apis" -> "A").
            segments = [seg for seg in re.split(r'[^A-Za-z0-9]+', core) if seg]
            return ''.join(seg[0] for seg in segments).upper() if segments else core[0].upper()

        def abbrev(n):
            m = re.search(r'(\d+)$', n)
            if m:
                return initials(n[:m.start()]) + m.group(1)
            # No trailing number (e.g. "drone", "robotA"): abbreviate to word initials,
            # keeping a trailing capital if present to distinguish same-base-name robots
            # (e.g. "robotA"/"robotB" -> "RA"/"RB").
            first = initials(n)
            if len(n) > 1 and n[-1].isupper():
                return first + n[-1]
            return first
        return '-'.join(abbrev(n) for n in names)
    
    # =========================================================================
    # ============================= Computation ===============================
    # =========================================================================
    @staticmethod
    def align_merged_trajectories(robot_names: List[str], est_data_lst: List[OdometryData], gt_data_lst: List[OdometryData]
                                  ) -> Tuple[PathData, PathData, List[PathData], List[PathData]]:
        """
        Merges and rigidly aligns per-robot estimated/ground-truth trajectories for a group of
        any size (including a single robot, i.e. self-alignment), without computing any error
        metrics. Shared by :meth:`calculate_merged_ate`, which computes metrics on top of this,
        and :meth:`save_merged_ate_figures`, which only needs the aligned trajectories to plot.

        Args:
            robot_names: Robot names in this group, in the same order as
                ``est_data_lst``/``gt_data_lst``.
            est_data_lst: Per-robot estimated trajectories, in ``robot_names`` order.
            gt_data_lst: Per-robot ground-truth trajectories, in ``robot_names`` order.

        Returns:
            Tuple ``(est_data_align, gt_data_align, est_data_align_list, gt_data_align_list)``:
            ``est_data_align``/``gt_data_align`` are the merged (all-robot), aligned/synced
            trajectories; ``est_data_align_list``/``gt_data_align_list`` are those same
            trajectories split back into their per-robot pieces, in ``robot_names`` order.
        """
        # Make the timestamps match and then merge (a single robot passes through as-is)
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
        est_data: PathData = PathData.concatenate_PathData(est_data_lst)
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # Time-sync and rigidly align the merged trajectories
        est_data_align, gt_data_align = PathData.align(gt_data, est_data, max_diff=0.1)

        # Split the aligned trajectories back into their single-robot forms.
        gt_data_align_list, est_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align, est_data_align)
        return est_data_align, gt_data_align, est_data_align_list, gt_data_align_list
        
    @staticmethod
    def calculate_merged_ate(slam_data: SLAMData,
                            load_gt_data_fn: Callable[[str, List[str]], List[OdometryData]],
                            rpe_delta: float = 5.0, rpe_delta_unit: Unit = Unit.meters) -> SLAMResult:
        """
        Compute the merged RMS ATE for a group of robots after ROMAN offline RPGO.

        Merges the estimated and ground-truth trajectories for every robot in the
        group (after aligning their time windows), then computes ATE on the
        combined trajectory. A single-robot group is a self-alignment case: no
        merging happens, and the "merged" trajectory is just that robot's own
        trajectory. Also computes the pre-optimize (first-stage) ATE when those
        trajectories were available. Trajectory / LC overlay plots are generated
        separately by :meth:`save_merged_ate_figures`.

        Args:
            slam_data: The loaded data for this run/robot group.
            load_gt_data_fn: Callable ``(dataset_seq, robot_names) -> List[OdometryData]``,
                dataset-specific.
            rpe_delta: Step size between the pose pairs used for all RPE calculations in
                this function (first-stage, merged, and per-robot). Does not affect ATE.
                Defaults to ``5.0``.
            rpe_delta_unit: Unit of ``rpe_delta``. Defaults to ``Unit.meters``, matching
                the fixed-distance RPE convention used by GrAco.

        Returns:
            SLAMResults
        """
        robot_names = slam_data.robot_names
        est_data_lst: List[OdometryData] = slam_data.estimated_trajectories
        gt_data_lst: List[OdometryData] = load_gt_data_fn(slam_data.system_params.dataset_version, robot_names)

        # Calculate first-stage (pre-optimize) metrics, when those trajectories were available
        first_stage_metrics = None
        if slam_data.pre_opt_est_trajectories:
            first_stage_est_lst, first_stage_gt_lst = PathData.make_start_and_end_times_match(
                slam_data.pre_opt_est_trajectories, gt_data_lst)
            first_stage_est: PathData = PathData.concatenate_PathData(first_stage_est_lst)
            first_stage_gt: PathData = PathData.concatenate_PathData(first_stage_gt_lst)
            first_stage_metrics, _, _ = OdometryData.align_and_calculate_traj_errors(first_stage_gt, first_stage_est, max_diff=0.1, visualize=False,
                                                                                    rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)

        # Merge and align trajectories
        est_data_align, gt_data_align, est_data_align_list, gt_data_align_list = \
                    SLAMEvaluator.align_merged_trajectories(robot_names, est_data_lst, gt_data_lst)
        
        # Calculate RMS ATE, among other metrics, on the merged (all-robot) trajectory
        metrics_dictionary = PathData.calculate_traj_errors(gt_data_align, est_data_align,
                                                            rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)

        # Compute each robot's individual post-optimize RMS ATE from its already-aligned pair
        robot_metrics = [
            PathData.calculate_traj_errors(gt_align, est_align, rpe_delta=rpe_delta, rpe_delta_unit=rpe_delta_unit)
            for gt_align, est_align in zip(gt_data_align_list, est_data_align_list)
        ]
        
        return SLAMResult(first_stage_metrics, metrics_dictionary, robot_metrics)

    # =========================================================================
    # ========================== Table Generation =============================
    # =========================================================================
    @staticmethod
    def _save_ate_tables(run_names: List[str], cols: List[str], multi_robot_cols: set,
                        run_display_names: Dict[str, str],
                        results: Dict[str, Dict[str, SLAMResult]],
                        save_path: Path, ate_threshold_m: float, rot_threshold_deg: float) -> None:
        """
        Build and save the ATE/RTE summary PDF tables.

        Produces five tables — pre-optimize (first-stage) merged RMS ATE,
        post-optimize merged RMS ATE, post-optimize merged RMS absolute rotation
        angle error, post-optimize merged RMS RTE, and post-optimize merged RMS
        relative rotation angle error — styled so that cells with no loop
        closures, or a value above ate_threshold_m (translation tables) / above
        rot_threshold_deg (rotation tables), are highlighted in red. The
        pre-optimize table is suppressed for multi-robot groups with zero total
        inter-robot LC; the post-optimize ATE, rotation error, RTE, and relative
        rotation error tables for multi-robot groups with zero inlier inter-robot
        LC (both via ``results[...].lc_stats_by_mode``/``lc_inlier_stats_by_mode``
        at ``LoopClosureFilterMode.ONLY_INTER_LC``). Single-robot (self-alignment) groups
        are never suppressed on LC grounds, since they have no inter-robot LC by
        definition.

        Each table gets a trailing "Average" column (the row-wise mean across
        the pair columns, ignoring suppressed/NaN pairs), set off from the pair
        columns by a heavy divider.

        Args:
            run_names: Ordered list of run identifiers (e.g. ``["ROMAN", "MG_TS"]``).
            cols: Ordered list of robot-pair column labels (e.g. ``["H1H2", "H1D1"]``).
            multi_robot_cols: Subset of ``cols`` whose group has more than one robot --
                only these are eligible for the zero-inter-robot-LC suppression.
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
                if result is None:
                    raise ValueError(
                        f"Missing results for run={run!r}, col={col!r} -- every (run, col) pair in "
                        "run_names/cols is expected to already be populated in results by this point.")
                val = metric_fn(result)
                no_inter_lc = False
                if col in multi_robot_cols:
                    lc_stats = lc_stats_selector(result)
                    if lc_stats is None:
                        raise ValueError(
                            f"Missing {LoopClosureFilterMode.ONLY_INTER_LC.name} LC stats for run={run!r}, col={col!r} -- "
                            "expected to always be populated by the LC-filter loop before this table is built.")
                    no_inter_lc = lc_stats['num_loop_closures'] == 0
                suppressed = val is None or no_inter_lc
                return float('nan') if suppressed else val
            raw_df = SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
            raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
            return raw_df

        inter_lc = lambda r: r.lc_stats_by_mode.get(LoopClosureFilterMode.ONLY_INTER_LC)
        inter_lc_inlier = lambda r: r.lc_inlier_stats_by_mode.get(LoopClosureFilterMode.ONLY_INTER_LC)

        color_fn_m = TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m)
        color_fn_deg = TableData.color_fn_NAVY_RED_missing_or_above(rot_threshold_deg)
        fmt = TableData.fmt_fixed(3)
        # The trailing "Average" column is a summary column, not another pair —
        # set it off from the pair columns with a heavy divider.
        heavy_divider_before = lambda col_idx: col_idx == len(cols)

        first_stage_ate_table = SLAMEvaluator.make_highlighted_table(
                    make_raw_df(lambda r: r.first_stage_metrics.APE.translation_part.rmse if r.first_stage_metrics else None, inter_lc),
                    "Merged RMS ATE (m) — Pre-Optimize",
                    color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
        ate_table = SLAMEvaluator.make_highlighted_table(
                    make_raw_df(lambda r: r.merged_metrics.APE.translation_part.rmse, inter_lc_inlier),
                    "Merged RMS ATE (m)",
                    color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
        rot_err_table = SLAMEvaluator.make_highlighted_table(
                    make_raw_df(lambda r: r.merged_metrics.APE.rotation_angle_deg.rmse, inter_lc_inlier),
                    "Merged RMS Absolute Rotation Error (deg)",
                    color_fn=color_fn_deg, fmt=fmt, higher_is_better=False)
        rte_table = SLAMEvaluator.make_highlighted_table(
                    make_raw_df(lambda r: r.merged_metrics.RPE.translation_part.rmse, inter_lc_inlier),
                    "Merged RMS RTE (m) - Δ5m",
                    color_fn=color_fn_m, fmt=fmt, higher_is_better=False)
        rel_rot_err_table = SLAMEvaluator.make_highlighted_table(
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

    @staticmethod
    def _save_ate_split_table(run_names: List[str], robot_groups: List[Tuple[str, ...]],
                            run_display_names: Dict[str, str],
                            results: Dict[str, Dict[str, SLAMResult]],
                            save_path: Path, ate_threshold_m: float) -> None:
        """
        Build and save the per-robot RMS ATE/RPE split summary PDF tables.

        For each robot group, produces one column per robot holding that robot's
        individual post-optimize RMS ATE/RPE, computed by separating the merged
        aligned trajectory back into per-robot trajectories (see
        :meth:`ROMANEvaluator.calculate_merged_ate`).

        Args:
            run_names: Ordered list of run identifiers.
            robot_groups: Ordered list of robot-name groups (each an arbitrary-length
                tuple/list), matching the group order used to build ``results``.
            run_display_names: Maps each run identifier to its display name in the table.
            results: ``DatasetSequenceResults`` keyed by run then column.
            save_path: Destination PDF path.
            ate_threshold_m: Red-highlight cutoff (m) for both tables (both are translation-only).
        """
        sub_cols = [f"{SLAMEvaluator.group_label(grp)}\n{name}" for grp in robot_groups for name in grp]

        subcol_group_idx = {f"{SLAMEvaluator.group_label(grp)}\n{name}": (SLAMEvaluator.group_label(grp), i)
                            for grp in robot_groups for i, name in enumerate(grp)}

        def make_raw_df(metric_fn) -> pd.DataFrame:
            def value_fn(run, subcol):
                group_lbl, i = subcol_group_idx[subcol]
                result = results[run].get(group_lbl)
                robot_metrics = result.robot_metrics[i] if result is not None else None
                return float('nan') if robot_metrics is None else metric_fn(robot_metrics)
            return SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: sub_cols, value_fn)

        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m)
        fmt = TableData.fmt_fixed(3)

        dfs = [
            SLAMEvaluator.make_highlighted_table(make_raw_df(lambda m: m.APE.translation_part.rmse), "Individual RMS ATE (m)",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
            SLAMEvaluator.make_highlighted_table(make_raw_df(lambda m: m.RPE.translation_part.rmse), "Individual RMS RPE (m) - Δ5m",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
        ]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, font_size=8, data_font_size=10)

    @staticmethod
    def save_merged_ate_figures(slam_data: SLAMData, method: str,
                                load_gt_data_fn: Callable[[str, List[str]], List[OdometryData]],
                                figures_base_dir: Path, viz_config: Dict) -> None:
        """
        Generate and save the 2D trajectory and LC-overlay PDFs for one run/group -- the
        visualization counterpart of :meth:`calculate_merged_ate`, split out so it can be run
        independently of ATE computation (e.g. sequentially, outside the parallel ``Pool`` used
        for metrics in :meth:`run_evaluation`).

        Aligns the trajectories the same way :meth:`calculate_merged_ate` does (via
        :meth:`align_merged_trajectories`), but computes no error metrics.

        Args:
            slam_data: The loaded data for this run/robot group.
            method: Run name, used for figure/file naming only.
            load_gt_data_fn: Callable ``(dataset_seq, robot_names) -> List[OdometryData]``,
                dataset-specific.
            figures_base_dir: Directory under which ``<dataset_name>/<dataset_seq>/traj`` and
                ``<dataset_name>/<dataset_seq>/<LoopClosureFilterMode.name>/traj_lc`` outputs are saved.
            viz_config: Dict with keys ``"image_path"``, ``"x_edge"``, ``"robot_name_to_color"``
                (keyed by display name), and optionally ``"name_map"`` (robot name -> display name;
                defaults to identity) and ``"yaw_rotation_deg"`` (rotates trajectories about the
                center of their combined bounding box before plotting against the background
                image; defaults to 0).
        """
        robot_names = slam_data.robot_names
        est_data_lst: List[OdometryData] = slam_data.estimated_trajectories
        gt_data_lst: List[OdometryData] = load_gt_data_fn(slam_data.system_params.dataset_version, robot_names)
        _, _, est_data_align_list, gt_data_align_list = \
            SLAMEvaluator.align_merged_trajectories(robot_names, est_data_lst, gt_data_lst)

        image_path = viz_config["image_path"]
        x_edge = viz_config["x_edge"]
        name_map: Dict = viz_config.get("name_map") or {rn: rn for rn in robot_names}
        robot_name_to_color: Dict = viz_config["robot_name_to_color"]
        image_extent_offsets = viz_config.get("background_image_extent_offsets")
        yaw_rotation_deg = viz_config.get("yaw_rotation_deg", 0.0)

        group_lbl = SLAMEvaluator.group_label(robot_names)
        base_dir = Path(figures_base_dir) / slam_data.system_params.dataset_name / slam_data.system_params.dataset_version
        traj_dir = base_dir / 'traj'
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Plot the results in 2D (Configuration for Figure 10) — LC-independent, saved once
        dataList  = [d for est, gt in zip(est_data_align_list, gt_data_align_list) for d in (est, gt)]
        isGTList  = [b for _ in robot_names for b in (False, True)]
        nameList  = [name_map[rn] for rn in robot_names for _ in range(2)]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True,
                        background_image_path=image_path, background_image_x_edge=x_edge,
                        background_image_extent_offsets=image_extent_offsets,
                        yaw_rotation_deg=yaw_rotation_deg,
                        save_path=str(traj_dir / f'traj_{group_lbl}_{method}.pdf'))

        # Plot only GT in 2D
        dataList  = gt_data_align_list
        isGTList  = [True] * len(robot_names)
        nameList  = [name_map[rn] for rn in robot_names]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=False,
                        background_image_path=image_path, background_image_x_edge=x_edge,
                        background_image_extent_offsets=image_extent_offsets,
                        gt_color_lightness_range_val=8,
                        yaw_rotation_deg=yaw_rotation_deg,
                        save_path=str(traj_dir / f'traj_{group_lbl}_{method}_onlyGT.pdf'))

        # Plot estimated trajectories with LC overlay (no background, no GT), once per LC filter mode.
        gt_dict_display = {name_map[rn]: gt for rn, gt in zip(robot_names, gt_data_lst)}
        est_dataList  = est_data_align_list
        est_isGTList  = [False] * len(robot_names)
        est_nameList  = [name_map[rn] for rn in robot_names]
        est_colorList = [robot_name_to_color[name] for name in est_nameList]
        for lc_filter in LoopClosureFilterMode:
            _, lc_data_inlier = slam_data.get_loop_closures(lc_filter)
            # SLAMData already resolved the g2o letters to robot names, so only the display
            # remap is left -- LC names must match est_nameList and gt_dict_display's keys.
            lc_data_inlier.apply_names_override(name_map)
            lc_data_inlier.calculate_errors(gt_dict_display)

            traj_lc_dir = base_dir / lc_filter.name / 'traj_lc'
            traj_lc_dir.mkdir(parents=True, exist_ok=True)
            PathData.visualize_2D(est_dataList, est_isGTList, est_colorList, est_nameList, no_background=True, line_width=1.0, show_grid=True,
                            loop_closure_data=lc_data_inlier, lc_line_width=2.0, lc_errors_vmax=2.0,
                            title=f"{method} LC overlaid on trajectory",
                            save_path=str(traj_lc_dir / f'traj_lc_{group_lbl}_{method}.pdf'))

    @staticmethod
    def _save_timing_table(run_names: List[str], cols: List[str],
                        run_display_names: Dict[str, str],
                        slam_data_by_run: Dict[str, Dict[str, SLAMData]],
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
            slam_data_by_run: Loaded SLAMData keyed by run then column.
            save_path: Destination PDF path.
        """
        def get_val(run: str, col: str, key: str):
            slam_data = slam_data_by_run[run].get(col)
            entry = SLAMData.get_timing_totals([slam_data]) if slam_data is not None else None
            if key == "total":
                return None if entry is None else entry["align"] + entry["mapping"] + entry["offline_rpgo"]
            return None if entry is None else entry[key]

        def make_raw_df(key: str) -> pd.DataFrame:
            def value_fn(run, col):
                v = get_val(run, col, key)
                return float('nan') if v is None else v
            raw_df = SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
            raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
            return raw_df

        style = TableData.TableStyleName.GEORGIA_TECH
        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'), style=style)
        fmt = TableData.fmt_fixed(1)
        # The trailing "Average" column is a summary column, not another pair —
        # set it off from the pair columns with a heavy divider.
        heavy_divider_before = lambda col_idx: col_idx == len(cols)

        dfs = [
            SLAMEvaluator.make_highlighted_table(make_raw_df("mapping"), "Mapping Runtime (s)",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
            SLAMEvaluator.make_highlighted_table(make_raw_df("align"), "Alignment Runtime (s)",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
            SLAMEvaluator.make_highlighted_table(make_raw_df("offline_rpgo"), "Offline RPGO Runtime (s)",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
            SLAMEvaluator.make_highlighted_table(make_raw_df("total"), "Total Runtime (s)",
                        color_fn=color_fn, fmt=fmt, higher_is_better=False),
        ]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, style=style,
                        heavy_divider_before=heavy_divider_before)

    @staticmethod
    def _save_data_size_table(run_names: List[str], cols: List[str],
                            run_display_names: Dict[str, str],
                            slam_data_by_run: Dict[str, Dict[str, SLAMData]],
                            save_path: Path) -> None:
        """
        Build and save the estimated communication data size summary PDF table.

        Also saves a standalone ``.tex`` version of the table (same path with a
        ``.tex`` suffix), ready to paste into Overleaf.

        Args:
            run_names: Ordered list of run identifiers.
            cols: Ordered list of robot-pair column labels.
            run_display_names: Maps each run identifier to its display name in the table.
            slam_data_by_run: Loaded SLAMData keyed by run then column.
            save_path: Destination PDF path.
        """
        def make_raw_df() -> pd.DataFrame:
            def value_fn(run, col):
                slam_data = slam_data_by_run[run].get(col)
                v = slam_data.data_size_mb if slam_data is not None else None
                return float('nan') if v is None else v
            raw_df = SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: cols, value_fn)
            raw_df["Average"] = raw_df.mean(axis=1, skipna=True)
            return raw_df

        style = TableData.TableStyleName.GEORGIA_TECH
        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'), style=style)
        fmt = TableData.fmt_fixed(2)
        # The trailing "Average" column is a summary column, not another pair —
        # set it off from the pair columns with a heavy divider.
        heavy_divider_before = lambda col_idx: col_idx == len(cols)

        data_size_table = SLAMEvaluator.make_highlighted_table(make_raw_df(), "Estimated Communication Data Size (MB)",
                            color_fn=color_fn, fmt=fmt, higher_is_better=False)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TableData.to_pdf([data_size_table], str(save_path), row_height=2.4, h_pad=0.5, style=style,
                        heavy_divider_before=heavy_divider_before)
        data_size_table.to_latex(str(save_path.with_suffix('.tex')),
                                caption="Estimated Communication Data Size (MB)", label="tab:data_size")

    @staticmethod
    def _save_lc_tables(run_names: List[str], run_display_names: Dict[str, str],
                        results: Dict[str, Dict[str, SLAMResult]],
                        lc_filter: LoopClosureFilterMode,
                        save_path: Path) -> None:
        """
        Build and save the LC summary PDF tables.

        Produces four tables: success rate and successful/total counts for all LC
        and for inlier LC, across all run names and robot pairs. The all-LC
        success rate table gets a trailing "Average" column (the row-wise mean
        across the pair columns, ignoring suppressed/NaN pairs), set off from the
        pair columns by a heavy divider — matching ``_save_ate_tables``.

        When ``lc_filter`` is ``LoopClosureFilterMode.ALL``, the all-LC success rate and
        successful/total tables are additionally saved as standalone ``.tex``
        files (``lc_success_rate_table.tex``, ``lc_successful_total_table.tex``)
        next to ``save_path``, ready to paste into Overleaf.

        Args:
            run_names: Ordered list of run identifiers.
            run_display_names: Maps each run identifier to its display name in the table.
            results: ``DatasetSequenceResults`` keyed by run then column; stats are
                read from ``.lc_stats_by_mode``/``.lc_inlier_stats_by_mode`` at ``lc_filter``.
            lc_filter: Which ``LoopClosureFilterMode`` to pull stats for.
            save_path: Destination PDF path.
        """
        def make_raw_df(stats_selector, key: str) -> pd.DataFrame:
            return SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: results[run].keys(),
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

        all_lc_success_rate_table = SLAMEvaluator.make_highlighted_table(make_success_rate_df(all_lc), "LC Success Rate %",
                    color_fn=color_fn, fmt=percent_fmt)
        all_lc_successful_total_table = SLAMEvaluator._style_combined_columns(make_raw_df(all_lc, "num_successful_loop_closures"),
                                make_raw_df(all_lc, "num_loop_closures"),
                                "LC Successful / Total", color_fn=color_fn, fmt=int_fmt)

        all_lc_success_rate_table_latex = SLAMEvaluator.make_highlighted_table(make_success_rate_df(all_lc), "LC Success Rate %",
                    color_fn=latex_color_fn, fmt=percent_fmt)
        all_lc_successful_total_table_latex = SLAMEvaluator._style_combined_columns(make_raw_df(all_lc, "num_successful_loop_closures"),
                                make_raw_df(all_lc, "num_loop_closures"),
                                "LC Successful / Total", color_fn=latex_color_fn, fmt=int_fmt)

        dfs = [
            all_lc_success_rate_table,
            all_lc_successful_total_table,
            SLAMEvaluator.make_highlighted_table(make_raw_df(inlier_lc, "success_rate"), "Inlier LC Success Rate %",
                        color_fn=color_fn, fmt=percent_fmt),
            SLAMEvaluator._style_combined_columns(make_raw_df(inlier_lc, "num_successful_loop_closures"),
                                    make_raw_df(inlier_lc, "num_loop_closures"),
                                    "Inlier LC Successful / Total", color_fn=color_fn, fmt=int_fmt),
        ]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TableData.to_pdf(dfs, str(save_path), row_height=2.4, h_pad=0.5, style=TableData.TableStyleName.GEORGIA_TECH,
                        heavy_divider_before=heavy_divider_before)

        if lc_filter == LoopClosureFilterMode.ALL:
            all_lc_success_rate_table_latex.to_latex(str(save_path.parent / 'lc_success_rate_table.tex'),
                                                caption="LC Success Rate \%", label="tab:lc_success_rate")
            all_lc_successful_total_table_latex.to_latex(str(save_path.parent / 'lc_successful_total_table.tex'),
                                                    caption="LC Successful / Total", label="tab:lc_successful_total")

    @staticmethod
    def _save_mg_match_histogram_grid_figure(run_names: List[str], run_display_names: Dict[str, str],
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

    @staticmethod
    def _save_mg_match_table(run_names: List[str], run_display_names: Dict[str, str], cols: List[str],
                            slam_data_by_run: Dict[str, Dict[str, SLAMData]],
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
            slam_data_by_run: Loaded SLAMData keyed by run then column; ``.mg_match``
                is ``None`` for pairs with no MG match files.
            save_path: Destination PDF path.
        """
        table_data_mg_match: Dict[str, Dict[str, Optional[Dict]]] = {
            run: {col: slam_data.mg_match for col, slam_data in slam_data_by_run[run].items()}
            for run in run_names
        }

        def make_raw_stage_count_df(stage: int) -> pd.DataFrame:
            def value_fn(run, col):
                stats = table_data_mg_match[run][col]
                return float('nan') if stats is None else float(stats['stage_counts'][stage])
            return SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: table_data_mg_match[run].keys(), value_fn)

        def make_raw_stage_percent_df(stage: int) -> pd.DataFrame:
            def value_fn(run, col):
                stats = table_data_mg_match[run][col]
                if stats is None:
                    return float('nan')
                counts = stats['stage_counts']
                total = counts[0] + counts[1] + counts[2]
                return float('nan') if total == 0 else 100 * counts[stage] / total
            return SLAMEvaluator._make_raw_df(run_names, run_display_names, lambda run: table_data_mg_match[run].keys(), value_fn)

        color_fn = TableData.color_fn_NAVY_RED_missing_or_above(float('inf'))
        int_fmt = TableData.fmt_fixed(0)
        percent_fmt = TableData.fmt_fixed(1, suffix='%')

        counts_table = SLAMEvaluator._style_combined_columns(make_raw_stage_count_df(1), make_raw_stage_count_df(2),
                                            "MG Match Stage Counts (1-2)",
                                            color_fn=color_fn, fmt=int_fmt, highlight=False, separator='-')
        percent_table = SLAMEvaluator._style_combined_columns(make_raw_stage_percent_df(1), make_raw_stage_percent_df(2),
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
                hist_fig = SLAMEvaluator._save_mg_match_histogram_grid_figure(run_names, run_display_names, cols,
                                                                table_data_mg_match, field, title, discrete)
                pp.savefig(hist_fig, bbox_inches='tight')
                plt.close(hist_fig)

    @staticmethod
    def _save_lc_context_figure(group: Tuple[str, ...], col: str,
                                    lc_data_list: List[LoopClosureData], labels_list: List[str],
                                    group_indices: List[int],
                                    stats_list: List[Dict],
                                    results: Dict[str, Dict[str, SLAMResult]],
                                    run_names: List[str], save_dir: Path, ate_threshold_m: float) -> None:
        """
        Generate and save a composite 16:9 slide figure for one robot group.

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
        match the PDF table titles produced by :meth:`ROMANEvaluator._save_ate_tables` and
        :meth:`ROMANEvaluator._save_lc_tables`.

        Args:
            group: Robot names identifying this group (e.g. ``("Husky1", "Drone1")``).
            col: Short group label used as the table column header and in the filename
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
                at ``LoopClosureFilterMode.ONLY_INTER_LC`` has zero loop closures, matching
                :meth:`ROMANEvaluator._save_ate_tables`.
            run_names: Ordered list of run identifiers.
            save_dir: Directory in which to save ``lc_context_<col>.pdf``.
            ate_threshold_m: Red-highlight cutoff (m) for the ATE table, matching :meth:`ROMANEvaluator._save_ate_tables`.
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
            successful_table = SLAMEvaluator.make_highlighted_table(pd.DataFrame({col_name: successful_series(idx)}), "Method", fmt=int_fmt)
            total_table = SLAMEvaluator.make_highlighted_table(pd.DataFrame({col_name: total_series(idx)}), "Method", fmt=int_fmt, emphasis_rankings=False)
            return TableData.merge_TableData(successful_table, total_table)

        # Combined tables: one per LC side, columns match PDF table titles
        table_l = SLAMEvaluator.make_highlighted_table(pd.DataFrame({COL_SR_ALL: sr_series(0)}), "Method", fmt=percent_fmt) \
            .append_TableData(make_combined_col(0, COL_CNT_ALL), axis=1)
        table_r = SLAMEvaluator.make_highlighted_table(pd.DataFrame({COL_SR_INL: sr_series(1)}), "Method", fmt=percent_fmt) \
            .append_TableData(make_combined_col(1, COL_CNT_INL), axis=1)

        def _ate_suppressed(rn: str) -> bool:
            result = results[rn].get(col)
            lc_stats = result.lc_inlier_stats_by_mode.get(LoopClosureFilterMode.ONLY_INTER_LC) if result is not None else None
            return (lc_stats or {}).get('num_loop_closures', -1) == 0

        def _ate(rn: str) -> Optional[float]:
            result = results[rn].get(col)
            return result.merged_metrics.APE.translation_part.rmse if result is not None else None

        ate_raw = pd.Series(
            {rn: float('nan') if _ate_suppressed(rn) or _ate(rn) is None else _ate(rn)
            for rn in run_names}
        ).reindex(run_names)

        table_ate = SLAMEvaluator.make_highlighted_table(
            pd.DataFrame({COL_ATE: ate_raw}), "Method",
            color_fn=TableData.color_fn_NAVY_RED_missing_or_above(ate_threshold_m), fmt=TableData.fmt_fixed(2), higher_is_better=False)

        # Scatter in center
        LoopClosureData.visualize_error_scatter(
            lc_data_list, labels_list, group_indices=group_indices,
            max_rotation_frac=1.0, max_translation_frac=1.0,
            show_plots=False, ax=ax_center)

        # Group name top-left in golden
        fig.text(0.01, 0.97, " / ".join(group),
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

    @staticmethod
    def _save_lc_side_by_side_figure(
        group: Tuple[str, ...],
        col: str,
        lc_data_list: List[LoopClosureData],
        labels_list: List[str],
        group_indices: List[int],
        run_names: List[str],
        save_dir: Path,
    ) -> None:
        """Generate a side-by-side LC scatter slide for one robot group.

        Produces a 22×12 inch figure with two equal scatter plots:
        - Left:  all loop closures plotted with X markers.
        - Right: inlier loop closures plotted with star markers.
        Both axes are forced square and share synchronized axis limits so errors
        are directly comparable. The group name is shown in the top-left corner.

        Args:
            group: Robot names identifying this group (e.g. ``("Husky1", "Drone1")``).
            col: Short group label used in the filename (e.g. ``"H1D1"``).
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

        fig.text(0.01, 0.97, " / ".join(group),
                fontsize=40, fontweight='bold', va='top', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)

        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_dir / f'lc_side_by_side_{col}.pdf'), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _save_lc_sep_figure(
        group: Tuple[str, ...],
        col: str,
        lc_data_list: List[LoopClosureData],
        labels_list: List[str],
        group_indices: List[int],
        run_names: List[str],
        run_display_names: Dict[str, str],
        save_dir: Path,
        inliers_only: bool = False,
    ) -> None:
        """Generate a per-method LC scatter slide for one robot group.

        Produces one square scatter panel per run, each showing only that run's
        loop closures, so overlapping methods never occlude one another. All
        panels share synchronized axis limits so errors are directly comparable.

        Args:
            group: Robot names identifying this group (e.g. ``("Husky1", "Drone1")``).
            col: Short group label used in the filename (e.g. ``"H1D1"``).
            lc_data_list: Interleaved list of all-LC and inlier-LC LoopClosureData for
                each run (length ``2 * len(run_names)``), each with ``calculate_errors``
                and ``label_successful`` already called. Even indices are all-LC; odd
                are inlier-LC.
            labels_list: Display label for each entry in ``lc_data_list``.
            group_indices: Group index per entry, following the pattern
                ``[0, 0, 1, 1, ..., n-1, n-1]``.
            run_names: Ordered list of run identifiers.
            run_display_names: Maps each run identifier to its display name (panel title).
            save_dir: Directory in which to save ``lc_sep_<col>.pdf``.
            inliers_only: If False (default), each panel shows all-LC as X markers and
                inlier-LC as star markers. If True, each panel shows only that run's
                inlier-LC as star markers, saved as ``lc_sep_inl_<col>.pdf``.
        """
        expected_group_indices = [i for i in range(len(run_names)) for _ in range(2)]
        assert group_indices == expected_group_indices, (
            f"group_indices must be interleaved pairs [0,0,1,1,...], "
            f"got {group_indices}, expected {expected_group_indices}")

        fig = plt.figure(figsize=(11 * len(run_names), 11))
        gs = gridspec.GridSpec(1, len(run_names), figure=fig,
                            left=0.035, right=0.965, bottom=0.08, top=0.92, wspace=0.25)
        axes = [fig.add_subplot(gs[i]) for i in range(len(run_names))]
        for ax in axes:
            ax.set_box_aspect(1)

        # Matches the palette visualize_error_scatter derives internally for the combined
        # 'lc' plot (group_indices=[0,0,1,1,...] there), so each method keeps its color
        run_palette = sns.color_palette("bright", len(run_names))
        for i, (ax, run_name) in enumerate(zip(axes, run_names)):
            if inliers_only:
                lc_inlier = lc_data_list[2 * i + 1]
                inlier_mask = np.ones(len(lc_inlier.results.translation_errors), dtype=bool)
                LoopClosureData.visualize_error_scatter(
                    [lc_inlier], [labels_list[2 * i + 1]],
                    inlier_masks=[inlier_mask], colors=[run_palette[i]],
                    max_rotation_frac=1.0, max_translation_frac=1.0,
                    show_plots=False, ax=ax)
            else:
                LoopClosureData.visualize_error_scatter(
                    lc_data_list[2 * i:2 * i + 2], labels_list[2 * i:2 * i + 2],
                    group_indices=[0, 0], colors=[run_palette[i], run_palette[i]],
                    max_rotation_frac=1.0, max_translation_frac=1.0,
                    show_plots=False, ax=ax)
            ax.set_title(run_display_names.get(run_name, run_name), fontsize=16, fontweight='bold')

        # Synchronize axis limits so errors are directly comparable
        x_min = min(ax.get_xlim()[0] for ax in axes)
        x_max = max(ax.get_xlim()[1] for ax in axes)
        y_min = min(ax.get_ylim()[0] for ax in axes)
        y_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig.text(0.01, 0.97, " / ".join(group),
                fontsize=40, fontweight='bold', va='top', color=TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).HeaderColor)

        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'lc_sep_inl_{col}.pdf' if inliers_only else f'lc_sep_{col}.pdf'
        fig.savefig(str(save_dir / filename), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _save_traj_lc_comb_figure(col: str, run_names: List[str], traj_lc_dir: Path, save_dir: Path) -> None:
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

    # =========================================================================
    # ============================= Evaluation ================================
    # =========================================================================

    @staticmethod
    def run_evaluation(mg_root: Path, dataset_name: str, dataset_seq: str, run_names: List[str],
                        robot_groups: List[Tuple[str, ...]],
                        critical_invocation_params: Dict[str, Any],
                        figures_base_dir: Path,
                        load_gt_data_fn: Callable[[str, List[str]], List[OdometryData]],
                        viz_config: Dict,
                        ate_threshold_m: float, rot_threshold_deg: float = 10.0) -> None:
        """
        Generate all evaluation figures and tables for one dataset.

        For each robot group across all run names:
        - Loads its :class:`SLAMData` and computes merged RMS ATE (pre- and post-optimize) in parallel.
        - For each ``LoopClosureFilterMode``, filters that group's loop closures,
            saves per-group LC error scatter plots (lc/) and success-rate plots
            (lc_success_rate/).
        - Saves a context figure combining the LC scatter with per-run stats and
            ATE for each group (lc_with_context/).

        Args:
            mg_root: Path to the MeronomyGraph repo checkout (with corresponding results).
            dataset_name: Result folder prefix identifying the dataset family (e.g. ``"hercules"``,
                ``"GrAco"``).
            dataset_seq: Dataset identifier (e.g. ``"V2.3.AC"``).
            run_names: Ordered list of run/method identifiers to evaluate.
            robot_groups: Explicit list of robot-name groups to evaluate, each an arbitrary-length
                tuple/list of names -- a singleton for self-alignment, a pair, a triplet, or the
                full robot set. Callers decide exactly what to plot (e.g.
                ``list(itertools.combinations(all_robots, 2))`` for every pairwise combination).
            critical_invocation_params: Other data-affecting args from the original run invocation.
            figures_base_dir: Directory under which ``figures/<dataset_name>/<dataset_seq>/`` outputs are saved.
            load_gt_data_fn: Callable ``(dataset_seq, robot_names) -> List[OdometryData]``,
                dataset-specific.
            viz_config: Dict forwarded to :meth:`ROMANEvaluator.save_merged_ate_figures` (see its docstring).
            ate_threshold_m: Red-highlight cutoff (m) for every translation-error table
                (ATE pre/post-optimize, individual ATE/RPE, RTE). Dataset-specific -- e.g.
                a smaller-area dataset like AirMuseum should use a smaller value than Hercules.
            rot_threshold_deg: Red-highlight cutoff (deg) for every rotation-error table
                (absolute and relative). Defaults to 10 degrees for all datasets.

        Outputs saved under ``figures/<dataset_name>/<dataset_seq>/``:
        - ``metrics_table.pdf``  — pre/post-optimize RMS ATE, absolute/relative rotation error, and RTE summary tables
        - ``ate_split_table.pdf`` — per-robot RMS ATE/RPE summary tables, one column per
            robot in each group
        - ``timing_table.pdf``   — alignment/offline RPGO/total runtime summary tables
        - ``data_size_table.pdf``, ``data_size_table.tex`` — estimated communication data size (MB) summary table
        - ``mg_match_table.pdf`` — MG two-stage matcher stage-count summary table
        - ``traj/``              — per-group estimated vs. GT trajectory plots

        Outputs saved under ``figures/<dataset_name>/<dataset_seq>/<LoopClosureFilterMode.name>/``, once per LC filter mode:
        - ``lc_tables.pdf``      — LC success rate and count summary tables
        - ``lc_success_rate_table.tex``, ``lc_successful_total_table.tex`` — under
            ``LoopClosureFilterMode.ALL`` only, standalone LaTeX versions of the all-LC
            success rate and successful/total tables
        - ``lc/<group>.pdf``     — per-group LC error scatter plots
        - ``lc_success_rate/``   — per-group LC success rate plots
        - ``lc_with_context/``   — per-group composite slide figures
        - ``lc_side_by_side/``   — per-group all-LC vs inlier-LC side-by-side scatter slides
        - ``lc_sep/``            — per-group, per-method LC scatter slides (one panel per run)
        - ``lc_sep_inl/``        — same as ``lc_sep/`` but showing only inlier LC per panel
        - ``traj_lc/``           — per-group estimated trajectory with LC overlay
        - ``traj_lc_comb/``      — per-group 2x2 combination of the per-method traj_lc slides
        """

        # results[run_name] is keyed by group_label(group) below; two different groups that
        # abbreviate to the same label would silently collide and overwrite each other's results
        # (e.g. "acl_jackal" and "acl_jackal2" both -> "A" before group_label's trailing-digit fix).
        # Fail loudly here instead of losing a column silently.
        cols_by_label: Dict[str, List[Tuple[str, ...]]] = {}
        for group in robot_groups:
            cols_by_label.setdefault(SLAMEvaluator.group_label(group), []).append(group)
        collisions = {label: groups for label, groups in cols_by_label.items() if len(groups) > 1}
        if collisions:
            raise ValueError(f"group_label collisions for {dataset_seq}: {collisions}")

        # Force the headless Agg backend regardless of whatever backend an earlier import may have
        # already selected -- safe here since this function never shows interactive figures (every
        # plot is saved via save_path), and avoids any attempt to open a display.
        matplotlib.use("Agg", force=True)

        # Define mapping between run name and display name
        run_display_names = {
            "ROMAN": "ROMAN (HERCULES replication)",
            "ROMAN_O": "ROMAN",
            "ROMAN_NM": "NM + ROMAN",
            "MG": "MeronomyGraph (Holonym Matching Only)",
            "MG_TS": "MeronomyGraph"
        }

        # Load every run/group's data, then compute its RMS ATE, both in parallel
        load_tasks = [(mg_root, dataset_name, dataset_seq, run_name, list(group), critical_invocation_params)
                for group in robot_groups
                for run_name in run_names]

        # Must happen here in the parent (not only inside the forked workers) so unpickling
        # MeronomyGraph-backed objects (e.g. SLAMData.system_params) back in the parent succeeds.
        SLAMData.ensure_MeronomyGraph_importable(mg_root)
        with Pool() as pool:
            loaded = pool.starmap(SLAMData.from_MeronomyGraph, load_tasks)
            pool_results = pool.starmap(SLAMEvaluator.calculate_merged_ate,
                                        [(slam_data, load_gt_data_fn) for slam_data in loaded])

        # All loaded data and computed results for this dataset, keyed by run then robot-group
        # column — the objects threaded through every table/figure function below.
        slam_data_by_run: Dict[str, Dict[str, SLAMData]] = {run: {} for run in run_names}
        results: Dict[str, Dict[str, SLAMResult]] = {run: {} for run in run_names}
        for (_, _, _, run_name, group, *_), slam_data, result in zip(load_tasks, loaded, pool_results):
            col = SLAMEvaluator.group_label(group)
            slam_data_by_run[run_name][col] = slam_data
            results[run_name][col] = result

        # Save trajectory/LC-overlay figures sequentially (not via Pool -- unlike ATE
        # computation, this touches matplotlib, which isn't safe to fan out across
        # worker processes with an interactive backend).
        for run_name in run_names:
            for col, slam_data in slam_data_by_run[run_name].items():
                SLAMEvaluator.save_merged_ate_figures(slam_data, run_name, load_gt_data_fn,
                                                      figures_base_dir, viz_config)

        # Define sequence group column names
        cols = [SLAMEvaluator.group_label(g) for g in robot_groups]

        base_dir = Path(figures_base_dir) / dataset_name / dataset_seq

        total_time_by_run = {}
        for run_name in run_names:
            # Deduped across this run's groups -- the pipeline caches, so mapping/alignment work
            # shared by several groups only ran once.
            total_time_by_run[run_name] = sum(
                SLAMData.get_timing_totals(list(slam_data_by_run[run_name].values())).values())
            print(f"{dataset_seq} {run_name}: total data generation time = {total_time_by_run[run_name]:.1f}s")
        print(f"{dataset_seq}: total data generation time across all runs = {sum(total_time_by_run.values()):.1f}s")

        SLAMEvaluator._save_timing_table(run_names, cols, run_display_names, slam_data_by_run, base_dir / 'timing_table.pdf')
        SLAMEvaluator._save_data_size_table(run_names, cols, run_display_names, slam_data_by_run, base_dir / 'data_size_table.pdf')
        SLAMEvaluator._save_mg_match_table(run_names, run_display_names, cols, slam_data_by_run, base_dir / 'mg_match_table.pdf')

        # Generate the LC-dependent outputs once per LC filter mode, each under its own subfolder.
        # Process ONLY_INTER_LC first so its inlier-LC stats are available for the ATE
        # suppression logic in _generate_lc_context_figure across all modes.
        for lc_filter in sorted(LoopClosureFilterMode, key=lambda m: m != LoopClosureFilterMode.ONLY_INTER_LC):
            mode_dir = base_dir / lc_filter.name
            subdirs = {name: mode_dir / name for name in
                    ('lc', 'lc_success_rate', 'lc_with_context', 'lc_side_by_side', 'lc_sep', 'lc_sep_inl', 'traj_lc', 'traj_lc_comb')}
            for subdir in subdirs.values():
                subdir.mkdir(parents=True, exist_ok=True)

            # For each group...
            for group in robot_groups:
                # Load GT Data
                col = SLAMEvaluator.group_label(group)
                gt_list = load_gt_data_fn(dataset_seq, list(group))
                gt_dict = {name: gt for name, gt in zip(group, gt_list)}

                # Calculate LC errors and visualize
                lc_data_list: List[LoopClosureData] = []
                labels_list: List[str] = []
                group_indices: List[int] = []
                for i, run_name in enumerate(run_names):
                    merged_lc, merged_lc_inlier = slam_data_by_run[run_name][col].get_loop_closures(lc_filter)
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

                SLAMEvaluator._save_lc_context_figure(group, col, lc_data_list, labels_list, group_indices,
                                            stats, results, run_names, subdirs['lc_with_context'], ate_threshold_m)
                SLAMEvaluator._save_lc_side_by_side_figure(group, col, lc_data_list, labels_list, group_indices,
                                                run_names, subdirs['lc_side_by_side'])
                SLAMEvaluator._save_lc_sep_figure(group, col, lc_data_list, labels_list, group_indices,
                                        run_names, run_display_names, subdirs['lc_sep'])
                SLAMEvaluator._save_lc_sep_figure(group, col, lc_data_list, labels_list, group_indices,
                                        run_names, run_display_names, subdirs['lc_sep_inl'], inliers_only=True)
                SLAMEvaluator._save_traj_lc_comb_figure(col, run_names, subdirs['traj_lc'], subdirs['traj_lc_comb'])

            SLAMEvaluator._save_lc_tables(run_names, run_display_names, results, lc_filter, mode_dir / 'lc_tables.pdf')

        # ATE table is LC-independent, so it's saved once at the dataset root. Cell suppression
        # (no LC present) is based on inter-robot LC only, since only inter-robot closures actually
        # connect the group's pose graph — intra-robot closures don't merge separate robots' trajectories.
        # Single-robot groups have no inter-robot LC by definition, so they're excluded from suppression.
        multi_robot_cols = {SLAMEvaluator.group_label(g) for g in robot_groups if len(g) > 1}
        SLAMEvaluator._save_ate_tables(run_names, cols, multi_robot_cols, run_display_names, results, base_dir / 'metrics_table.pdf',
                        ate_threshold_m, rot_threshold_deg)

        # Per-robot RMS ATE/RPE split, also LC-independent and saved once at the dataset root.
        SLAMEvaluator._save_ate_split_table(run_names, robot_groups, run_display_names, results,
                            base_dir / 'ate_split_table.pdf', ate_threshold_m)

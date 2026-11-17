from dataclasses import dataclass, field
from robotdataprocess import LoopClosureFilterMode, PathData, PathDataAlignResult
from typing import Dict, List, Optional

@dataclass
class SLAMResult:
    """
    All computed results for one run (method) on one robot group within a dataset. Holds only
    what evaluation computes -- the data it was computed from lives on the run's ``SLAMData``.

    Populated incrementally: :meth:`SLAMEvaluator.calculate_merged_ate` fills in the
    trajectory-error fields; the per-``LoopClosureFilterMode`` loop-closure stats are set
    afterward in :meth:`SLAMEvaluator.run_evaluation`.

    Attributes:
        first_stage_metrics: Pre-optimize trajectory error metrics on the merged
            (all-robot) trajectory, or None if the pre-optimize file was unavailable.
        merged_metrics: Post-optimize trajectory error metrics on the merged trajectory.
        robot_metrics: Post-optimize trajectory error metrics for each robot in the
            group, in the same order as the group's robot names, computed by
            separating the merged aligned trajectory back apart.
        est_align_list: Post-optimize estimated trajectory of each robot, aligned onto GT
            (see :meth:`SLAMEvaluator.align_merged_trajectories`), in the group's robot name order.
        gt_align_list: GT trajectory of each robot, in the group's robot name order. Not the full
            GT: it's cropped to only the poses time-matched (within 0.1 s) to ``est_align_list``,
            after :meth:`PathData.make_start_and_end_times_match` pads the endpoints.
        lc_stats_by_mode: All-LC stats (see
            :meth:`LoopClosureData.visualize_error_scatter`), keyed by ``LoopClosureFilterMode``.
        lc_inlier_stats_by_mode: Inlier-LC stats, keyed by ``LoopClosureFilterMode``.
    """
    first_stage_metrics: Optional[PathDataAlignResult]
    merged_metrics: PathDataAlignResult
    robot_metrics: List[PathDataAlignResult]
    est_align_list: List[PathData]
    gt_align_list: List[PathData]
    lc_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)
    lc_inlier_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)

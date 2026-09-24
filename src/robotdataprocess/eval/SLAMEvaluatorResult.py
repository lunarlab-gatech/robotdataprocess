from dataclasses import dataclass, field
from robotdataprocess import LoopClosureFilterMode, PathDataAlignResult
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
        lc_stats_by_mode: All-LC stats (see
            :meth:`LoopClosureData.visualize_error_scatter`), keyed by ``LoopClosureFilterMode``.
        lc_inlier_stats_by_mode: Inlier-LC stats, keyed by ``LoopClosureFilterMode``.
    """
    first_stage_metrics: Optional[PathDataAlignResult]
    merged_metrics: PathDataAlignResult
    robot_metrics: List[PathDataAlignResult]
    lc_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)
    lc_inlier_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)

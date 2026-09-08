from dataclasses import dataclass, field
from robotdataprocess import LoopClosureFilterMode, PathDataAlignResult
from typing import Dict, List, Optional

@dataclass
class SLAMResult:
    """
    All computed results for one run (method) on one robot group within a dataset.

    Populated incrementally: :meth:`SLAMEvaluator.calculate_merged_ate` fills in the
    trajectory-error fields; ``timing``/``data_size_mb``/``mg_match`` and the
    per-``LoopClosureFilterMode`` loop-closure stats are set afterward as each is
    computed in :meth:`SLAMEvaluator.run_evaluation`.

    Attributes:
        first_stage_metrics: Pre-optimize trajectory error metrics on the merged
            (all-robot) trajectory, or None if the pre-optimize file was unavailable.
        merged_metrics: Post-optimize trajectory error metrics on the merged trajectory.
        robot_metrics: Post-optimize trajectory error metrics for each robot in the
            group, in the same order as the group's robot names, computed by
            separating the merged aligned trajectory back apart.
        timing: Runtime breakdown (``{"align": seconds, "mapping": seconds, "offline_rpgo": seconds}``),
            or None if the runtime files were unavailable.
        data_size_mb: Estimated communication data size in decimal MB, or None
            if the data size file was unavailable.
        mg_match: MG two-stage matcher stage-count/field stats (see
            :meth:`SLAMEvaluator.load_mg_match_stats`), or None if not an MG run or no
            match files exist.
        lc_stats_by_mode: All-LC stats (see
            :meth:`LoopClosureData.visualize_error_scatter`), keyed by ``LoopClosureFilterMode``.
        lc_inlier_stats_by_mode: Inlier-LC stats, keyed by ``LoopClosureFilterMode``.
    """
    first_stage_metrics: Optional[PathDataAlignResult]
    merged_metrics: PathDataAlignResult
    robot_metrics: List[PathDataAlignResult]
    timing: Optional[Dict[str, float]] = None
    data_size_mb: Optional[float] = None
    mg_match: Optional[Dict] = None
    lc_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)
    lc_inlier_stats_by_mode: Dict[LoopClosureFilterMode, Dict] = field(default_factory=dict)

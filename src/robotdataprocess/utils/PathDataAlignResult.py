from typeguard import typechecked
from typing import Optional

@typechecked
class TrajErrorStatistics:
    """
    Summary statistics for a single trajectory error metric (e.g. APE
    translation error), as computed by evo.

    Attributes:
        rmse: Root mean square error.
        mean: Mean error.
        median: Median error.
        std: Standard deviation of the error.
        min: Minimum error.
        max: Maximum error.
        sse: Sum of squared errors.
    """

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    sse: float

    @typechecked
    def __init__(self, rmse: float, mean: float, median: float, std: float, min: float, max: float, sse: float):
        self.rmse = rmse
        self.mean = mean
        self.median = median
        self.std = std
        self.min = min
        self.max = max
        self.sse = sse

@typechecked
class PoseRelationErrors:
    """
    Error statistics broken down by evo ``PoseRelation``, for a single
    metric (APE or RPE).

    Attributes:
        full_transformation: Error over the full SE(3) transformation, dimensionless.
        translation_part: Translational error, in meters.
        rotation_part: Rotational error (as a rotation matrix component), dimensionless.
        rotation_angle_deg: Rotational error, in degrees.
        rotation_angle_rad: Rotational error, in radians.
        point_distance: Point distance error, in meters.
        point_distance_error_ratio: TODO.
    """

    full_transformation: TrajErrorStatistics
    translation_part: TrajErrorStatistics
    rotation_part: TrajErrorStatistics
    rotation_angle_deg: TrajErrorStatistics
    rotation_angle_rad: TrajErrorStatistics
    point_distance: TrajErrorStatistics
    point_distance_error_ratio: Optional[TrajErrorStatistics]

    @typechecked
    def __init__(self, full_transformation: TrajErrorStatistics, translation_part: TrajErrorStatistics,
                 rotation_part: TrajErrorStatistics, rotation_angle_deg: TrajErrorStatistics,
                 rotation_angle_rad: TrajErrorStatistics, point_distance: TrajErrorStatistics,
                 point_distance_error_ratio: Optional[TrajErrorStatistics] = None):
        self.full_transformation = full_transformation
        self.translation_part = translation_part
        self.rotation_part = rotation_part
        self.rotation_angle_deg = rotation_angle_deg
        self.rotation_angle_rad = rotation_angle_rad
        self.point_distance = point_distance
        self.point_distance_error_ratio = point_distance_error_ratio

@typechecked
class PathDataAlignResult:
    """
    Result of :meth:`PathData.calculate_traj_errors` /
    :meth:`PathData.align_and_calculate_traj_errors`: trajectory error
    metrics computed between a ground truth and an aligned estimated
    trajectory, via the evo library.

    Attributes:
        APE: Absolute Pose Error statistics, by ``PoseRelation``.
        RPE: Relative Pose Error statistics, by ``PoseRelation``.
    """

    APE: PoseRelationErrors
    RPE: PoseRelationErrors

    @typechecked
    def __init__(self, APE: PoseRelationErrors, RPE: PoseRelationErrors):
        self.APE = APE
        self.RPE = RPE

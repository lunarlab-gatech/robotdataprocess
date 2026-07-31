from typeguard import typechecked
from typing import Optional
import numpy as np

class LoopClosureDataResult:
    """
    Per-loop-closure error metrics for a LoopClosureData instance, as computed
    by :meth:`LoopClosureData.calculate_errors`.

    Attributes:
        translation_errors: (N,) float64 array of translation magnitude errors in meters.
        rotation_errors: (N,) float64 array of rotation angle errors in degrees.
        successful: (N,) boolean array marking which loop closures fall within the
            thresholds passed to :meth:`label_successful`, or None until that
            method has been called.
        trans_err_in_target: Translation error threshold (m) passed to
            :meth:`label_successful`, or None until that method has been called.
        rot_err_in_target: Rotation error threshold (deg) passed to
            :meth:`label_successful`, or None until that method has been called.
    """

    translation_errors: np.ndarray
    rotation_errors: np.ndarray
    successful: Optional[np.ndarray]
    trans_err_in_target: Optional[float]
    rot_err_in_target: Optional[float]

    @typechecked
    def __init__(self, translation_errors: np.ndarray, rotation_errors: np.ndarray):
        self.translation_errors = translation_errors
        self.rotation_errors = rotation_errors
        self.successful = None
        self.trans_err_in_target = None
        self.rot_err_in_target = None

    @typechecked
    def label_successful(self, trans_err_in_target: float, rot_err_in_target: float) -> None:
        """
        Labels each loop closure as successful if its translation and rotation
        errors are both within the given thresholds. Sets ``self.successful``,
        ``self.trans_err_in_target``, and ``self.rot_err_in_target``.
        """
        self.trans_err_in_target = trans_err_in_target
        self.rot_err_in_target = rot_err_in_target
        self.successful = (self.translation_errors <= trans_err_in_target) & (self.rotation_errors <= rot_err_in_target)

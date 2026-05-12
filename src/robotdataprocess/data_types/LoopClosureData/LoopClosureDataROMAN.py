from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import List, Tuple, Union

from .LoopClosureData import LoopClosureData


class LoopClosureDataROMAN(LoopClosureData):
    """
    LoopClosureData subclass for loop closures produced by the ROMAN/CLIPPER
    pipeline. Extends the base class with per-loop-closure CLIPPER association
    statistics that are written into the JSON output by that method.
    """

    clipper_num_associations: np.ndarray  # (N,) int array, optional
    n_reg_corrs: np.ndarray  # (N,) int array, optional
    n_syn_corrs: np.ndarray  # (N,) int array, optional
    n_overlapping_init: np.ndarray  # (N,) int array, optional

    @typechecked
    def __init__(self, timestamps_a: Union[np.ndarray, list], timestamps_b: Union[np.ndarray, list],
                 names: List[Tuple[str, str]], translations: Union[np.ndarray, list], orientations: Union[np.ndarray, list],
                 detected_inliers: Union[np.ndarray, list, None] = None,
                 clipper_num_associations: Union[np.ndarray, list, None] = None,
                 n_reg_corrs: Union[np.ndarray, list, None] = None,
                 n_syn_corrs: Union[np.ndarray, list, None] = None,
                 n_overlapping_init: Union[np.ndarray, list, None] = None):

        super().__init__(timestamps_a, timestamps_b, names, translations, orientations, detected_inliers)
        self.clipper_num_associations = np.array(clipper_num_associations, dtype=np.int64) if clipper_num_associations is not None else None
        self.n_reg_corrs = np.array(n_reg_corrs, dtype=np.int64) if n_reg_corrs is not None else None
        self.n_syn_corrs = np.array(n_syn_corrs, dtype=np.int64) if n_syn_corrs is not None else None
        self.n_overlapping_init = np.array(n_overlapping_init, dtype=np.int64) if n_overlapping_init is not None else None

    @classmethod
    def from_json(cls, json_path: Union[Path, str],
                  names_override: Union[dict, None] = None) -> LoopClosureDataROMAN:
        """
        Creates a LoopClosureDataROMAN instance from a JSON file. Delegates
        base field parsing to :meth:`LoopClosureData.from_json`, then reads
        the CLIPPER-specific fields from the same file.

        Args:
            json_path: Path to the JSON file.
            names_override: Optional dict mapping names found in the JSON to
                desired replacement names. Passed through to the base parser.

        Returns:
            LoopClosureDataROMAN instance.
        """
        base = LoopClosureData.from_json(json_path, names_override)

        with open(str(json_path), 'r') as f:
            data = json.load(f)

        clipper_num_associations = [entry.get("clipper_num_associations") for entry in data]
        n_reg_corrs = [entry.get("n_reg_corrs") for entry in data]
        n_syn_corrs = [entry.get("n_syn_corrs") for entry in data]
        n_overlapping_init = [entry.get("n_overlapping_init") for entry in data]

        def _opt_arr(lst):
            return np.array(lst, dtype=np.int64) if any(v is not None for v in lst) else None

        return cls(
            timestamps_a=base.timestamps_a,
            timestamps_b=base.timestamps_b,
            names=base.names,
            translations=base.translations,
            orientations=base.orientations,
            detected_inliers=getattr(base, 'detected_inliers', None),
            clipper_num_associations=_opt_arr(clipper_num_associations),
            n_reg_corrs=_opt_arr(n_reg_corrs),
            n_syn_corrs=_opt_arr(n_syn_corrs),
            n_overlapping_init=_opt_arr(n_overlapping_init),
        )

    # =========================================================================
    # ===================== Multi LoopClosureData Methods =====================
    # =========================================================================

    @staticmethod
    def merge(loop_closures: List) -> LoopClosureDataROMAN:
        raise NotImplementedError

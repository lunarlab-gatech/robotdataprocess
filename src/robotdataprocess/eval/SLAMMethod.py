from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SLAMMethod:
    """
    One method to evaluate, defined by the parameters its results were generated with: an experiment
    config (``base_param_config``) plus any command-line param overrides applied on top of it, which
    together determine where its results live.

    Attributes:
        name: Unique method identifier, used as the method's key and in figure filenames.
        display_name: Method name shown in tables, legends, and figure titles.
        base_param_config: Experiment config the results were generated with, with ``param_overrides``
            applied on top.
        color: Hex color of the method in the ``all_methods`` figures.
        param_overrides: Dot-notation param overrides applied on top of ``base_param_config`` (e.g.
            ``{"submap_align_params.submap_max_size": 35}``), or ``None`` for none.
    """
    name: str
    display_name: str
    base_param_config: str
    color: str
    param_overrides: Optional[Dict[str, Any]] = None

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RobotGroup:
    """
    One fully-resolved robot group to evaluate in :meth:`SLAMEvaluator.run_evaluation`. A plain
    ``Tuple[str, ...]``/``List[str]`` of robot names is also accepted there and normalized into
    one of these.

    Attributes:
        robots: Robot names in this group, in any order.
        dataset_seq: Dataset sequence this group's data comes from.
        label: Column/figure-filename label for this group.
    """
    robots: Tuple[str, ...]
    dataset_seq: str
    label: str

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RobotGroup:
    """
    A group of robots for a specific dataset and sequence.

    Attributes:
        robots: Robot names in this group, in any order.
        dataset_name: Result folder prefix identifying the dataset family (e.g. ``"hercules"``).
        dataset_seq: Dataset sequence this group's data comes from.
        label: Column/figure-filename label for this group.
    """
    robots: Tuple[str, ...]
    dataset_name: str
    dataset_seq: str
    label: str

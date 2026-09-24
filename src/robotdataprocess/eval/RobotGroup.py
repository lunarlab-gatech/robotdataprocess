from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RobotGroupViz:
    """
    Background-image config for one :class:`RobotGroup`'s figures in
    :meth:`SLAMEvaluator.save_merged_ate_figures`.

    Attributes:
        name_map: Maps each robot name to its display name, or ``None`` for identity.
        robot_name_to_color: Maps each display name (post ``name_map``) to a hex color.
        image_path: Path to the background environment image, or ``None`` for no background.
        image_x_edge: Background image's real-world width, used to scale it against the trajectories.
        image_extent_offsets: ``(x, y)`` offsets applied to the background image's extent.
        yaw_rotation_deg: Rotates trajectories about their combined bounding box center before
            plotting against the background image.
    """
    name_map: Optional[Dict[str, str]]
    robot_name_to_color: Dict[str, str]
    image_path: Optional[str]
    image_x_edge: Optional[float]
    image_extent_offsets: Optional[Tuple[float, float]]
    yaw_rotation_deg: float = 0.0


@dataclass(frozen=True)
class RobotGroup:
    """
    A group of robots for a specific dataset and sequence.

    Attributes:
        robots: Robot names in this group, in any order.
        dataset_name: Result folder prefix identifying the dataset family (e.g. ``"hercules"``).
        dataset_seq: Dataset sequence this group's data comes from.
        label: Column/figure-filename label for this group.
        viz_config: Background-image config for this group's figures.
    """
    robots: Tuple[str, ...]
    dataset_name: str
    dataset_seq: str
    label: str
    viz_config: RobotGroupViz

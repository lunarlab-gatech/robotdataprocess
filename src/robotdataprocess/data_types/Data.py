from __future__ import annotations

from enum import Enum
from typeguard import typechecked

class CoordinateFrame(Enum):
    """
    Enum for different coordinate frames used in robotics.

    Attributes:
        FLU: X forward, Y left, Z up := RHS
        NED: X forward (north), Y right (east), Z down := RHS
        ENU: X right (east), Y forward (north), Z up := RHS
        NONE: No defined coordinate frame.
    """

    FLU = 0
    NED = 1
    ENU = 2
    NONE = 3

class TransformType(Enum):
    """
    Enum for how coordinate frame conversions are applied to a transformation.

    Attributes:
        ROTATION: Apply the frame change as a rotation (left-multiply).
            q_new = q_frame_change * q_old, t_new = R_frame_change * t_old.
        CHANGE_OF_BASIS: Apply the frame change as a similarity transform.
            T_new = R * T * R^{-1}, where R is the frame change matrix.
    """

    ROTATION = 0
    CHANGE_OF_BASIS = 1

class ROSMsgLibType(Enum):
    """
    Enum for different ROS message library types.

    Attributes:
        ROSBAGS: Use ROS messages from the rosbags library (Pure Python library).
        RCLPY: Use ROS messages from the rclpy library (ROS2 Python client library).
        ROSPY: Use ROS messages from the rospy library (ROS1 Python client library).
        NONE: No ROS message library (for testing purposes only).
    """

    ROSBAGS = 0
    RCLPY = 1
    ROSPY = 2
    NONE = 3

class Data:
    """
    Minimal base class for all data types. Provides a shared frame_id attribute.
    """

    frame_id: str

    @typechecked
    def __init__(self, frame_id: str):
        self.frame_id = frame_id

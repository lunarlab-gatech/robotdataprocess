from __future__ import annotations

from enum import Enum
import numpy as np
from typeguard import typechecked

class CoordinateFrame(Enum):
    """
    Enum for different coordinate frames used in robotics.

    Attributes:
        FLU:               X forward,            Y left,    Z up := RHS
        NED (FRD): X forward (north),    Y right (east),  Z down := RHS
        ENU (RFU):      right (east), Y forward (north),    Z up := RHS
        FUR:               X forward,              Y up, Z right := RHS
        UFL:                    X up,         Y forward,  Z left := RHS
        NONE: No defined coordinate frame.
    """

    FLU = 0
    NED = 1
    ENU = 2
    FUR = 3
    UFL = 4
    NONE = 5

    @staticmethod
    def _axis_vector(letter: str) -> list:
        """
        Maps a single axis letter to its unit vector, canonical to FLU
        (forward := +X, left := +Y, up := +Z). Compass letters (used by e.g.
        NED, ENU) alias to their body-relative equivalent.
        """
        aliases = {'N': 'F', 'S': 'B', 'E': 'R', 'W': 'L'}
        vectors = {
            'F': [1, 0, 0], 'B': [-1, 0, 0],
            'L': [0, 1, 0], 'R': [0, -1, 0],
            'U': [0, 0, 1], 'D': [0, 0, -1],
        }

        letter = aliases.get(letter, letter)
        if letter not in vectors:
            raise ValueError(f"Unrecognized coordinate frame axis letter '{letter}'.")
        return vectors[letter]

    def _axes_matrix(self) -> np.ndarray:
        return np.array([CoordinateFrame._axis_vector(letter) for letter in self.name]).T

    @staticmethod
    def get_rotation(src_frame: CoordinateFrame, dst_frame: CoordinateFrame) -> np.ndarray:
        """
        Computes the rotation matrix converting vectors expressed in
        ``src_frame`` to vectors expressed in ``dst_frame``, derived from
        each frame's name (e.g. NED -> FLU is 180 degrees about X).

        Args:
            src_frame: The coordinate frame the input is expressed in.
            dst_frame: The coordinate frame the output should be expressed in.

        Returns:
            3x3 rotation matrix R such that v_dst = R @ v_src.
        """
        if src_frame == CoordinateFrame.NONE or dst_frame == CoordinateFrame.NONE:
            raise ValueError("Cannot compute a rotation to/from CoordinateFrame.NONE.")

        return dst_frame._axes_matrix().T @ src_frame._axes_matrix()

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

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        if self.frame_id != other.frame_id:
            print(f"  [__eq__] frame_id: {self.frame_id!r} != {other.frame_id!r}")
            return False
        return True

from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from decimal import Decimal
from enum import Enum
import numpy as np
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

class ROSMsgLibType(Enum):
    """ 
    Enum for different ROS message library types.
     
    Attributes:
        ROSBAGS: Use ROS messages from the rosbags library (Pure Python library).
        RCLPY: Use ROS messages from the rclpy library (ROS2 Python client library).
        ROSPY: Use ROS messages from the rospy library (ROS1 Python client library).
    """

    ROSBAGS = 0
    RCLPY = 1
    ROSPY = 2

class Data:
    """
    Generic Data class that provides example methods that should be overwritten by children,
    and can run operations between different data types.
    """

    # Define data attributes for all Data classes
    frame_id: str
    timestamps: np.ndarray[Decimal]

    @typechecked
    def __init__(self, frame_id: str, timestamps: np.ndarray | list):
        
        # Copy initial values into attributes
        self.frame_id = frame_id
        self.timestamps = col_to_dec_arr(timestamps)

        # Check to ensure that all timestamps are sequential
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] >= self.timestamps[i+1]:
                raise ValueError(f"Timestamps {self.timestamps[i]} and {self.timestamps[i+1]} do not come in sequential order!")
            
    def len(self):
        """ Returns the number of items in this data class """
        return len(self.timestamps)
    
    @staticmethod
    def get_ros_msg_type(self, libtype: ROSMsgLibType):
        """ Will return the msgtype for the ROS message for this Data object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def get_ros_msg(self, libtype: ROSMsgLibType, i: int):
        """ Will create and return a ROS message object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")


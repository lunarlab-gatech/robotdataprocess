from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from decimal import Decimal
from enum import Enum
import numpy as np
from typeguard import typechecked

class CoordinateFrame(Enum):
    FLU = 0 # https://www.ros.org/reps/rep-0103.html - X forward, Y left, Z up := RHS
    NED = 1 # https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
            # - X forward (north), Y right (East), Z Down := RHS
    ENU = 2 # https://www.ros.org/reps/rep-0103.html - X east, Y north, Z up := RHS
    NONE = 3 # No defined coordinate frame

class Data:
    """
    Generic Data class that provides example methods that should be overwritten by children,
    and can run operations between different data types.
    """

    # Define data attributes for all Data classes
    frame_id: str
    timestamps: np.ndarray[Decimal]

    @typechecked
    def __init__(self, frame_id: str, timestamps: np.ndarray | list, ):
        
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
    def get_ros_msg_type():
        """ Will return the msgtype needed to add a connetion to a rosbag writer. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def get_ros_msg(self, i: int):
        """ Will return a ROS message object ready to be written into a ROS2 Humble bag. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")


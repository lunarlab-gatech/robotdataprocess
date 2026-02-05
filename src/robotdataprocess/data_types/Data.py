from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from decimal import Decimal
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import List

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
        NONE: No ROS message library (for testing purposes only).
    """

    ROSBAGS = 0
    RCLPY = 1
    ROSPY = 2
    NONE = 3

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
    def get_ros_msg_type(libtype: ROSMsgLibType):
        """ Will return the msgtype for the ROS message for this Data object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def get_ros_msg(self, libtype: ROSMsgLibType, i: int):
        """ Will create and return a ROS message object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    def hertz_analysis(self) -> None:
        """ Plot historgrams with the sequential data hertz rates and time differences. """

        # Calculate the differences between consecutive timestamps
        if self.len() < 2:
            raise ValueError(f"Not enough data samples to analyze hertz.")
        hertz_diffs = [self.timestamps[i] - self.timestamps[i - 1] for i in range(1, self.len())]
        hertz_values = [1 / diff for diff in hertz_diffs if diff > 0]

        # Output a message if some differences are zero
        num_seq_messages_with_same_timestamps = len([x for x in hertz_diffs if x == 0])
        if num_seq_messages_with_same_timestamps > 0:
            print(f"Warning: Sequential Pairs of timestamps are equivalent {num_seq_messages_with_same_timestamps} times.")

        # Sort each of these Lists
        hertz_diffs.sort()
        hertz_values.sort()

        # To remove potentially noisy values, remove first and last 5
        if len(hertz_diffs) > 5:
            hertz_diffs = hertz_diffs[5:-5]
            hertz_values = hertz_values[5:-5]

        # Create histograms for the hertz values and time differences
        def create_histogram(data: List, title: str, xlabel: str, ylabel: str) -> None:
            """ Create and show a histogram from the provided data. """

            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=100, color='blue', alpha=0.7)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.grid(True)
            plt.yscale('log')
            plt.show()

        create_histogram(
            data=hertz_diffs,
            title=f'Time Differences for {self.__class__.__name__}',
            xlabel='Time Difference (seconds)',
            ylabel='Seq. Message Pairs Count (#)'
        )
        create_histogram(
            data=hertz_values,
            title=f'Hertz Analysis for {self.__class__.__name__}',
            xlabel='Hertz (Hz)',
            ylabel='Seq. Message Pairs Count (#)',
        )


from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from .Data import Data, CoordinateFrame, ROSMsgLibType
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from typeguard import typechecked
from typing import List

class SequentialData(Data):
    """
    Data class for sequential (time-ordered) data. Provides timestamps, hertz analysis,
    and methods that should be overwritten by children.
    """

    timestamps: np.ndarray[Decimal]

    @typechecked
    def __init__(self, frame_id: str, timestamps: np.ndarray | list):

        super().__init__(frame_id)
        self.timestamps = col_to_dec_arr(timestamps)

        # Check to ensure that all timestamps are sequential
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] >= self.timestamps[i+1]:
                raise ValueError(f"Timestamps {self.timestamps[i]} and {self.timestamps[i+1]} do not come in sequential order!")

    def len(self):
        """ Returns the number of items in this data class """
        return len(self.timestamps)

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(libtype: ROSMsgLibType):
        """ Will return the msgtype for the ROS message for this Data object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    def get_ros_msg(self, libtype: ROSMsgLibType, i: int):
        """ Will create and return a ROS message object. """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    # =========================================================================
    # ============================ Data Analysis ==============================
    # =========================================================================

    def compute_hertz_stats(self, trim_outliers: bool = True) -> tuple[List, List, int]:
        """
        Compute hertz statistics from timestamps.

        Args:
            trim_outliers: If True and there are more than 5 samples, remove first and last 5 values.

        Returns:
            Tuple of (hertz_diffs, hertz_values, num_zero_diffs) where:
                - hertz_diffs: List of time differences between consecutive timestamps
                - hertz_values: List of hertz values (1/diff) for non-zero differences
                - num_zero_diffs: Number of consecutive timestamp pairs with zero difference

        Raises:
            ValueError: If there are fewer than 2 data samples.
        """
        if self.len() < 2:
            raise ValueError(f"Not enough data samples to analyze hertz.")

        hertz_diffs = [self.timestamps[i] - self.timestamps[i - 1] for i in range(1, self.len())]
        hertz_values = [1 / diff for diff in hertz_diffs if diff > 0]
        num_zero_diffs = len([x for x in hertz_diffs if x == 0])

        # Sort each of these Lists
        hertz_diffs.sort()
        hertz_values.sort()

        # To remove potentially noisy values, remove first and last 5
        if trim_outliers:
            if len(hertz_diffs) <= 10:
                raise ValueError(f"Not enough data to trim outliers. Need more than 10 differences, got {len(hertz_diffs)}.")
            hertz_diffs = hertz_diffs[5:-5]
            hertz_values = hertz_values[5:-5]

        return hertz_diffs, hertz_values, num_zero_diffs

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    def hertz_analysis(self, show_plots: bool = True) -> tuple[List, List]:
        """
        Plot histograms with the sequential data hertz rates and time differences.

        Args:
            show_plots: If True, display matplotlib plots. Set to False for testing.

        Returns:
            Tuple of (hertz_diffs, hertz_values) for testing purposes.
        """
        hertz_diffs, hertz_values, num_zero_diffs = self.compute_hertz_stats()

        # Output a message if some differences are zero
        if num_zero_diffs > 0:
            print(f"Warning: Sequential Pairs of timestamps are equivalent {num_zero_diffs} times.")

        if show_plots:
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

        return hertz_diffs, hertz_values

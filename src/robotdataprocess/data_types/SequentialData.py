from __future__ import annotations

from ..utils.conversion_utils import col_to_dec_arr
from .Data import Data, CoordinateFrame, ROSMsgLibType
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from typeguard import typechecked
from typing import List, Tuple

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
        warned = False
        for i in range(len(self.timestamps) - 1):
            if not warned and self.timestamps[i] >= self.timestamps[i+1]:
                print(f"Warning: Timestamps {self.timestamps[i]} and {self.timestamps[i+1]} do not come in sequential order!")
                warned = True

    def __eq__(self, other) -> bool:
        parent_result = super().__eq__(other)
        if parent_result is not True:
            return parent_result
        if not np.array_equal(self.timestamps, other.timestamps):
            if self.timestamps.shape != other.timestamps.shape:
                print(f"  [__eq__] timestamps shape: {self.timestamps.shape} != {other.timestamps.shape}")
            else:
                idx = next(i for i in range(len(self.timestamps)) if self.timestamps[i] != other.timestamps[i])
                print(f"  [__eq__] timestamps first diff at idx {idx}: {self.timestamps[idx]} != {other.timestamps[idx]}")
            return False
        return True

    def len(self) -> int:
        """
        Returns the number of items in this data class.

        Returns:
            int: The number of timestamped entries.
        """
        return len(self.timestamps)

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in SequentialData. """
        pass

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def round_timestamps(self, decimals: int):
        """
        Rounds all timestamps to the specified number of decimal places.

        Args:
            decimals: Number of decimal places to round to.
        """
        quantize_val = Decimal(10) ** -decimals
        self.timestamps = np.array([ts.quantize(quantize_val) for ts in self.timestamps])
        self._invalidate_cache()

    def crop_data(self, start: Decimal, end: Decimal):
        """
        Will crop the data so only values within [start, end] inclusive are kept.

        Args:
            start: The earliest timestamp to keep.
            end: The latest timestamp to keep.

        Raises:
            NotImplementedError: Always; must be overridden by subclasses.
        """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(libtype: ROSMsgLibType):
        """
        Will return the msgtype for the ROS message for this Data object.

        Args:
            libtype: Which ROS message library to use.

        Returns:
            The ROS message type class.

        Raises:
            NotImplementedError: Always; must be overridden by subclasses.
        """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    def get_ros_msg(self, libtype: ROSMsgLibType, i: int):
        """
        Will create and return a ROS message object.

        Args:
            libtype: Which ROS message library to use.
            i: Index of the data sample to convert.

        Returns:
            A ROS message populated with the data at index ``i``.

        Raises:
            NotImplementedError: Always; must be overridden by subclasses.
        """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")

    # =========================================================================
    # ============================ Data Analysis ==============================
    # =========================================================================

    def get_rate_hz(self) -> float:
        """
        Computes this sequential data's average rate, from its first to last timestamp.

        Returns:
            The average rate, in Hz.

        Raises:
            ValueError: If there are fewer than 2 data samples.
        """
        if self.len() < 2:
            raise ValueError(f"Not enough data samples to compute rate.")
        duration = float(self.timestamps[-1] - self.timestamps[0])
        return (self.len() - 1) / duration

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

    # =========================================================================
    # ===================== Multi SequentialData Methods =======================
    # =========================================================================

    @staticmethod
    def _compute_matched_indices(timestamps1: np.ndarray, timestamps2: np.ndarray,
                                 tolerance: Decimal) -> Tuple[List[int], List[int]]:
        """
        Find index pairs ``(i, j)`` such that
        ``abs(timestamps1[i] - timestamps2[j]) <= tolerance``, with each
        timestamp matched to at most one timestamp in the other array
        (one-to-one).

        Assumes both arrays are sorted. Matching uses a greedy two-pointer
        sweep: at each step, if the current pair of timestamps is within
        ``tolerance`` they are matched and both pointers advance; otherwise,
        whichever pointer is at the earlier timestamp advances alone.

        Args:
            timestamps1: The first array of (sorted) timestamps.
            timestamps2: The second array of (sorted) timestamps.
            tolerance: Maximum allowed absolute time difference between
                matched timestamps.

        Returns:
            Tuple of ``(matched_indices1, matched_indices2)``, parallel lists
            of matched indices into ``timestamps1`` and ``timestamps2``.
        """

        i, j = 0, 0
        matched_i: List[int] = []
        matched_j: List[int] = []
        while i < len(timestamps1) and j < len(timestamps2):
            diff = timestamps1[i] - timestamps2[j]
            if abs(diff) <= tolerance:
                matched_i.append(i)
                matched_j.append(j)
                i += 1
                j += 1
            elif diff < 0:
                i += 1
            else:
                j += 1

        return matched_i, matched_j

    @staticmethod
    def crop_to_matched(data1: SequentialData, data2: SequentialData,
                       tolerance: Decimal) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop two SequentialData objects in place so only mutually-matched
        entries remain: for each kept pair of indices ``(i, j)``,
        ``abs(data1.timestamps[i] - data2.timestamps[j]) <= tolerance``, and
        each timestamp is matched to at most one timestamp in the other
        object (one-to-one).

        Assumes both objects' timestamps are sorted (as ``__init__`` already
        expects). See ``_compute_matched_indices`` for the matching algorithm.

        Subclasses with additional per-timestamp arrays (e.g. images, poses,
        points) should call this method to crop ``timestamps``, then apply
        the returned masks to their own arrays.

        Args:
            data1: The first SequentialData object, cropped in place.
            data2: The second SequentialData object, cropped in place.
            tolerance: Maximum allowed absolute time difference between
                matched timestamps.

        Returns:
            Tuple of ``(mask1, mask2)``: boolean masks, relative to each
            object's original (pre-crop) timestamps, indicating which
            entries were kept.
        """

        matched_i, matched_j = SequentialData._compute_matched_indices(
            data1.timestamps, data2.timestamps, tolerance)

        mask1 = np.zeros(len(data1.timestamps), dtype=bool)
        mask1[matched_i] = True
        mask2 = np.zeros(len(data2.timestamps), dtype=bool)
        mask2[matched_j] = True

        data1.timestamps = data1.timestamps[mask1]
        data2.timestamps = data2.timestamps[mask2]
        data1._invalidate_cache()
        data2._invalidate_cache()

        return mask1, mask2

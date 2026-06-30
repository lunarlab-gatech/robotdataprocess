from __future__ import annotations

import colorsys
from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
import copy
import csv
from .Data import CoordinateFrame, TransformType
from .SequentialData import SequentialData
from decimal import Decimal
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
import math
from ..math_utils import interpolate_poses
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import pandas as pd
from pathlib import Path
from ..ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
from typing import Dict, Union, Tuple, List, Optional
import tqdm

@typechecked
class PathData(SequentialData):
    """
    Trajectory data with timestamped 3D positions and orientations.

    Extends SequentialData with spatial pose information. Serves as the base
    class for OdometryData and provides methods for frame conversion, trajectory
    alignment, error evaluation (APE/RPE via evo), and 2D/3D visualization.

    Attributes:
        positions: (N, 3) array of x, y, z positions in meters.
        orientations: (N, 4) array of quaternions in (x, y, z, w) order.
        frame: The coordinate frame convention of this data.
    """

    positions: np.ndarray # meters (x, y, z)
    orientations: np.ndarray # quaternions (x, y, z, w)
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, List], 
                 positions: Union[np.ndarray, List], orientations: Union[np.ndarray, List], 
                 frame: CoordinateFrame):
        super().__init__(frame_id, timestamps)
        self.positions = col_to_dec_arr(positions)
        self.orientations = col_to_dec_arr(orientations)
        self.frame = frame

    def __eq__(self, other) -> bool:
        parent_result = super().__eq__(other)
        if parent_result is not True:
            return parent_result
        if not np.array_equal(self.positions, other.positions):
            if self.positions.shape != other.positions.shape:
                print(f"  [__eq__] positions shape: {self.positions.shape} != {other.positions.shape}")
            else:
                idx = next(i for i in range(len(self.positions)) if not np.array_equal(self.positions[i], other.positions[i]))
                print(f"  [__eq__] positions first diff at idx {idx}: {self.positions[idx]} != {other.positions[idx]}")
            return False
        if not np.array_equal(self.orientations, other.orientations):
            if self.orientations.shape != other.orientations.shape:
                print(f"  [__eq__] orientations shape: {self.orientations.shape} != {other.orientations.shape}")
            else:
                idx = next(i for i in range(len(self.orientations)) if not np.array_equal(self.orientations[i], other.orientations[i]))
                print(f"  [__eq__] orientations first diff at idx {idx}: {self.orientations[idx]} != {other.orientations[idx]}")
            return False
        if self.frame != other.frame:
            print(f"  [__eq__] frame: {self.frame} != {other.frame}")
            return False
        return True

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in PathData. """
        pass

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def crop_data(self, start: Decimal, end: Union[Decimal, None] = None):
        """
        Will crop the data so only values within [start, end] inclusive are kept.

        Args:
            start: The earliest timestamp to keep.
            end: The latest timestamp to keep. If None, keeps all data from ``start`` onward.
        """

        # Create boolean mask of data to keep
        mask = ((self.timestamps >= start) & (self.timestamps <= end)) if end is not None else (self.timestamps >= start)

        # Apply mask
        self.timestamps = self.timestamps[mask]
        self.positions = self.positions[mask]
        self.orientations = self.orientations[mask]

        self._invalidate_cache()

    def shift_position(self, x_shift: float, y_shift: float, z_shift: float):
        """
        Shifts the positions of the path.

        Args:
            x_shift (float): Shift in x-axis.
            y_shift (float): Shift in y_axis.
            z_shift (float): Shift in z_axis.
        """
        self.positions[:,0] += Decimal(x_shift)
        self.positions[:,1] += Decimal(y_shift)
        self.positions[:,2] += Decimal(z_shift)

        self._invalidate_cache()

    def shift_to_start_at_identity(self):
        """
        Alter the positions and orientations based so that the first pose
        starts at Identity.
        """

        # Get pose of first robot position w.r.t world
        R_o = R.from_quat(self.orientations[0]).as_matrix()
        T_o = np.expand_dims(self.positions[0], axis=1)

        # Calculate the inverse (pose of world w.r.t first robot location)
        R_inv = R_o.T

        # Rotate positions and orientations
        self.positions = (R_inv @ (self.positions.T - T_o).astype(float)).T
        for i in range(self.len()):
            self.orientations[i] = R.from_matrix((R_inv @ R.from_quat(self.orientations[i]).as_matrix())).as_quat()

        # Convert back to decimal array
        self.positions = col_to_dec_arr(self.positions)
        self.orientations = col_to_dec_arr(self.orientations)

        self._invalidate_cache()

    def interpolate_to_hz(self, target_hz: float):
        """
        Linearly interpolates position and SLERPs orientation so that path data
        is output at the desired frequency.

        Args:
            target_hz (float): Desired output frequency in Hertz (e.g. 6.0)
        """

        # Check that target_hz is valid
        if target_hz <= 0: raise ValueError("target_hz must be positive")

        # Convert timestamps to float seconds
        ts = dec_arr_to_float_arr(self.timestamps)
        pos = dec_arr_to_float_arr(self.positions)
        quat = dec_arr_to_float_arr(self.orientations)

        # Create new evenly spaced timestamps
        duration = ts[-1] - ts[0]
        num_samples = int(np.ceil(duration * target_hz)) + 1
        new_ts = np.linspace(ts[0], ts[-1], num_samples)

        # Interpolate poses
        new_pos, new_quat = interpolate_poses(ts, pos, quat, new_ts)

        # Convert back to Decimal arrays
        self.timestamps = col_to_dec_arr(new_ts)
        self.positions = col_to_dec_arr(new_pos)
        self.orientations = col_to_dec_arr(new_quat)

        self._invalidate_cache()

    # =========================================================================
    # =========================== Frame Conversions ===========================
    # =========================================================================

    def to_coordinate_frame(self, target_frame: CoordinateFrame, transform_type: TransformType = TransformType.CHANGE_OF_BASIS):
        """
        Converts positions and orientations into the target coordinate frame.

        Args:
            target_frame: The desired coordinate frame.
            transform_type: How to apply the frame change.
                ``CHANGE_OF_BASIS`` applies a similarity transform to orientations
                (R * q * R^-1) and rotates positions (default, original behaviour).
                ``ROTATION`` left-multiplies the frame change rotation onto both
                positions and orientations without the inverse on orientations.
        """
        if self.frame == target_frame:
            print(f"Data already in {target_frame.name} coordinate frame, returning...")
            return

        if self.frame == CoordinateFrame.NED and target_frame == CoordinateFrame.FLU:
            # The frame change rotation: 180 degrees around X
            R_frame = np.array([[1,  0,  0],
                                [0, -1,  0],
                                [0,  0, -1]])

            if transform_type == TransformType.CHANGE_OF_BASIS:
                self._convert_frame(R_frame)
            elif transform_type == TransformType.ROTATION:
                R_frame_Q = R.from_matrix(R_frame)
                self.positions = col_to_dec_arr((R_frame @ self.positions.T).T)
                self._ori_apply_rotation(R_frame_Q)

            self.frame = CoordinateFrame.FLU
            self._invalidate_cache()

        else:
            raise NotImplementedError(f"Transformation from {self.frame} to {target_frame} is not implemented.")

    def apply_transformation_left_side(self, H: np.ndarray):
        """
        Applies a rigid-body transformation to the entire path.
        In terms of transformation matrices, this multiplies this path
        on the left side.

        Args:
            H: The 4x4 transformation matrix
        """

        # Apply the transformation
        H_self = np.eye(4).reshape(1, 4, 4).repeat(self.len(), axis=0)  # shape (N,4,4)
        H_self[:, :3, :3] = R.from_quat(self.orientations).as_matrix()
        H_self[:, :3, 3] = self.positions
        H_output = H @ H_self

        # Extract results and save
        self.positions = H_output [:, :3, 3]
        self.orientations = R.from_matrix(H_output[:, :3, :3]).as_quat()

        self._invalidate_cache()

    def apply_transformation_right_side(self, H: np.ndarray):
        """
        Applies a rigid-body transformation to the entire path.
        In terms of transformation matrices, this multiplies this path
        on the right side (row-vector convention).

        Args:
            H: The 4x4 transformation matrix
        """

        # Apply the transformation
        H_self = np.eye(4).reshape(1, 4, 4).repeat(self.len(), axis=0)  # shape (N,4,4)
        H_self[:, :3, :3] = R.from_quat(self.orientations).as_matrix()
        H_self[:, :3, 3] = self.positions
        H_output = H_self @ H

        # Extract results and save
        self.positions = H_output [:, :3, 3]
        self.orientations = R.from_matrix(H_output[:, :3, :3]).as_quat()

        self._invalidate_cache()

    def _convert_frame(self, R_frame: np.ndarray):
        """ Uses a change of basis to update the positions and orientations. """
        R_frame_Q = R.from_matrix(R_frame)
        self.positions = col_to_dec_arr((R_frame @ self.positions.T).T)
        self._ori_change_of_basis(R_frame_Q)

        self._invalidate_cache()

    def _ori_apply_rotation(self, R_i: R):
        """ Applies a rotation (not a change of basis) to orientations, thus stays in the same frame. """
        for i in range(self.len()):
            self.orientations[i] = (R_i * R.from_quat(self.orientations[i])).as_quat()
        self.orientations = col_to_dec_arr(self.orientations)

        self._invalidate_cache()

    def _ori_change_of_basis(self, R_i: R):
        """ Applies a change of basis to orientations """
        for i in range(self.len()):
            self.orientations[i] = (R_i * R.from_quat(self.orientations[i]) * R_i.inv()).as_quat()
        self.orientations = col_to_dec_arr(self.orientations)

        self._invalidate_cache()

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_ros2_bag(cls, bag_path: Union[Path, str], odom_topic: str, frame: CoordinateFrame) -> PathData:
        """
        Creates a class structure from a ROS2 bag file with a Path topic.

        Args:
            bag_path (Union[Path, str]): Path to the ROS2 bag file.
            odom_topic (str): Topic of the Path messages.
            frame: The coordinate frame of this data.
        Returns:
            PathData: Instance of this class.
        """

        # Get topic message count and typestore
        bag_wrapper = Ros2BagWrapper(bag_path, None)
        typestore: Typestore = bag_wrapper.get_typestore()
        num_msgs: int = bag_wrapper.get_topic_count(odom_topic)
        
        # Make empty arrays
        timestamps_np = np.zeros(0, dtype=Decimal)
        positions_np = np.zeros((0, 3), dtype=Decimal)
        orientations_np = np.zeros((0, 4), dtype=Decimal)

        # Setup tqdm bar & counter
        pbar = tqdm.tqdm(total=num_msgs, desc="Extracting Path...", unit=" msgs")

        # Extract the odometry information
        frame_id = None
        with Reader2(str(bag_path)) as reader:

            # Extract frame_id from first message
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):  
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                frame_id = msg.header.frame_id
                break

            # Extract message data
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                
                # NOTE: Currently, this method doesn't track when each Path message 
                # is recieved, and throws away duplicate poses contained in multiple
                # Path messages.

                # Iterate through each pose in the message
                for pose in msg.poses:
                    
                    # See if we already have this pose (via timestamp)
                    ts = bag_wrapper.extract_timestamp(pose)
                    if ts in timestamps_np:
                        continue

                    # If not, extract data
                    timestamps_np = np.concatenate((timestamps_np, [ts]), axis= 0)
                    pos = pose.pose.position
                    positions_np = np.concatenate((positions_np, [[Decimal(pos.x), Decimal(pos.y), Decimal(pos.z)]]), axis=0)
                    ori = pose.pose.orientation
                    orientations_np = np.concatenate((orientations_np, [[Decimal(ori.x), Decimal(ori.y), Decimal(ori.z), Decimal(ori.w)]]), axis=0)

                    # Increment the count
                    pbar.update(1)

        # Create an OdometryData class
        return cls(frame_id, timestamps_np, positions_np, orientations_np, frame)
    
    @classmethod
    def from_evo(cls, pose_trajectory_3d: PoseTrajectory3D, frame_id: str, frame: CoordinateFrame) -> PathData:
        """
        Creates a PathData object from an evo PoseTrajectory3D object.

        Args:
            pose_trajectory_3d: An evo PoseTrajectory3D with positions, orientations, and timestamps.
            frame_id: The frame ID to assign.
            frame: The coordinate frame of this data.

        Returns:
            PathData: Instance of this class.
        """

        # Convert orientations from wxyz to xyzw
        orientations_xyzw = pose_trajectory_3d.orientations_quat_wxyz[:, [1, 2, 3, 0]]

        # Remove duplicate timestamps (can arise after sync.associate_trajectories)
        timestamps = pose_trajectory_3d.timestamps
        duplicate_mask = np.concatenate(([False], np.diff(timestamps) == 0))
        if np.any(duplicate_mask):
            dup_indices = np.where(duplicate_mask)[0]
            for i in dup_indices:
                if not (np.allclose(pose_trajectory_3d.positions_xyz[i], pose_trajectory_3d.positions_xyz[i - 1], atol=1e-9, rtol=0) and
                        np.allclose(orientations_xyzw[i], orientations_xyzw[i - 1], atol=1e-9, rtol=0)):
                    raise ValueError(f"Duplicate timestamp {timestamps[i]} at index {i} has mismatched position or orientation.")
        unique_mask = ~duplicate_mask
        timestamps = timestamps[unique_mask]
        positions = pose_trajectory_3d.positions_xyz[unique_mask]
        orientations_xyzw = orientations_xyzw[unique_mask]

        return cls(frame_id=frame_id,
                   timestamps=timestamps,
                   positions=positions,
                   orientations=orientations_xyzw,
                   frame=frame)

    @classmethod
    def from_csv(cls, csv_path: Union[Path, str], frame_id: str, frame: CoordinateFrame,
                 header_included: bool, column_to_data: Union[List[int], None] = None,
                 separator: Union[str, None] = None, filter: Union[Tuple[str, str], None] = None,
                 ts_in_ns: bool = False, reorder_data: bool = False):
        """
        Creates a class structure from a csv file.

        Args:
            csv_path (Path | str): Path to the CSV file.
            frame_id (str): The frame that this path is relative to.
            frame (CoordinateFrame): The coordinate system convention of this data.
            header_included (bool): If this csv file has a header, so we can remove it.
            column_to_data (list[int]): Tells the algorithms which columns in the csv contain which
                of the following data: ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']. Thus,
                index 0 of column_to_data should be the column that timestamp data is found in the
                csv file. Set to None to use [0,1,2,3,4,5,6,7].
            separator (str | None): The separator used in the csv file. If None, will use a comma by default.
            filter: A tuple of (column_name, value), where only rows with column_name == value will be kept. If
                csv file has no headers, then `column_name` should be the index of the column as a string.
            ts_in_ns (bool): If True, assumes timestamps are in nanoseconds and converts to seconds. Otherwise,
                assumes timestamps are already in seconds.
            reorder_data (bool): If True, reorders the data to be in order of timestamps. If False,
                assumes data is already ordered by timestamp.

        Returns:
            PathData: Instance of this class.
        """

        # If column_to_data is None, assume default
        if column_to_data is None:
            column_to_data = [0,1,2,3,4,5,6,7]
        else:
            # Check column_to_data values are valid
            assert np.all(np.array(column_to_data) >= 0)
            assert len(column_to_data) == 8

        # Read the csv file
        header = 0 if header_included else None
        df1 = pd.read_csv(str(csv_path), header=header, index_col=False, sep=separator, engine='python')

        # Rename columns to standard names
        rename_dict = {}
        desired_data = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        for j, ind in enumerate(column_to_data):
            rename_dict[df1.columns[ind]] = desired_data[j]
        df1 = df1.rename(columns=rename_dict)

        # Using the filter if provided, remove unwanted rows
        if filter is not None:
            df1 = df1[df1[filter[0]] == filter[1]]

        # Convert columns to NDArray[Decimal]
        timestamps_np = np.array([Decimal(str(ts)) for ts in df1['timestamp']], dtype=object)
        positions_np = np.array([[Decimal(str(x)), Decimal(str(y)), Decimal(str(z))]
                                 for x, y, z in zip(df1['x'], df1['y'], df1['z'])], dtype=object)
        orientations_np = np.array([[Decimal(str(qx)), Decimal(str(qy)), Decimal(str(qz)), Decimal(str(qw))]
                                    for qx, qy, qz, qw in zip(df1['qx'], df1['qy'], df1['qz'], df1['qw'])], dtype=object)

        # If timestamps are in ns, convert to s
        if ts_in_ns:
            timestamps_np = timestamps_np / Decimal('1e9')

        # Reorder the data if needed
        if reorder_data:
            print("Warning: This code is not tested yet!")
            sort_indices = np.argsort(timestamps_np)
            timestamps_np = timestamps_np[sort_indices]
            positions_np = positions_np[sort_indices]
            orientations_np = orientations_np[sort_indices]

        # Create a PathData instance
        return PathData(frame_id, timestamps_np, positions_np, orientations_np, frame)

    @classmethod
    def from_txt(cls, file_path: Union[Path, str], frame_id: str, frame: CoordinateFrame,
                 header_included: bool, column_to_data: Union[List[int], None] = None):
        """
        Creates a PathData class from a text file.

        Args:
            file_path (Path | str): Path to the file containing the path data.
            frame_id (str): The frame where this path is relative to.
            frame (CoordinateFrame): The coordinate system convention of this data.
            header_included (bool): If this text file has a header, so we can remove it.
            column_to_data (list[int]): Tells the algorithms which columns in the text file contain which
                of the following data: ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']. Thus,
                index 0 of column_to_data should be the column that timestamp data is found in the
                text file. Set to None to use [0,1,2,3,4,5,6,7].
        Returns:
            PathData: Instance of this class.
        """

        # If column_to_data is None, assume default
        if column_to_data is None:
            column_to_data = [0,1,2,3,4,5,6,7]
        else:
            # Check column_to_data values are valid
            assert np.all(np.array(column_to_data) >= 0)
            assert len(column_to_data) == 8

        # Count the number of lines in the file
        line_count = 0
        with open(str(file_path), 'r') as file:
            for _ in file:
                line_count += 1

        # Setup arrays to hold data
        timestamps_np = np.zeros((line_count), dtype=object)
        positions_np = np.zeros((line_count, 3), dtype=object)
        orientations_np = np.zeros((line_count, 4), dtype=object)

        # Open the txt file and read in the data
        with open(str(file_path), 'r') as file:
            for i, line in enumerate(file):
                line_split = line.split(' ')
                timestamps_np[i] = line_split[column_to_data[0]]
                positions_np[i] = np.array([line_split[column_to_data[1]], line_split[column_to_data[2]], line_split[column_to_data[3]]])
                orientations_np[i] = np.array([line_split[column_to_data[5]], line_split[column_to_data[6]], line_split[column_to_data[7]], line_split[column_to_data[4]]])

        # Remove the header
        if header_included:
            timestamps_np = timestamps_np[1:]
            positions_np = positions_np[1:]
            orientations_np = orientations_np[1:]

        # Create a PathData instance
        return PathData(frame_id, timestamps_np, positions_np, orientations_np, frame)

    @classmethod
    def from_tum(cls, file_path: Union[Path, str], frame_id: str, frame: CoordinateFrame) -> PathData:
        """
        Creates a PathData class from a TUM RGB-D dataset trajectory format text file.

        Each row must contain 8 space-separated values::

            timestamp x y z q_x q_y q_z q_w

        where ``timestamp`` is in seconds and the orientation quaternion is in
        ``(x, y, z, w)`` order.

        Args:
            file_path (Path | str): Path to the TUM trajectory file.
            frame_id (str): The frame where this path is relative to.
            frame (CoordinateFrame): The coordinate system convention of this data.

        Returns:
            PathData: Instance of this class.
        """
        # TUM order: ts x y z qx qy qz qw
        # column_to_data: ts=0, x=1, y=2, z=3, qw=7, qx=4, qy=5, qz=6
        return cls.from_txt(file_path, frame_id, frame,
                            header_included=False,
                            column_to_data=[0, 1, 2, 3, 7, 4, 5, 6])

    @classmethod
    def from_g2o(cls, g2o_path: Union[Path, str], time_path: Union[Path, str],
                 robot: str, frame_id: str, frame: CoordinateFrame,
                 names_override: Union[dict, None] = None) -> PathData:
        """
        Creates a PathData instance from a g2o file containing VERTEX_SE3:QUAT entries.

        Only VERTEX_SE3:QUAT entries are read; EDGE_SE3:QUAT entries are ignored.
        GTSAM symbol keys are decoded as ``character = chr(key >> 56)`` and
        ``index = key & ((1 << 56) - 1)``. If ``names_override`` is provided,
        character keys are remapped before matching against ``robot``.

        The g2o quaternion order is (qx, qy, qz, qw), which matches the xyzw
        convention used by this class.

        Args:
            g2o_path: Path to the .g2o file.
            time_path: Path to a timestamp file. Each line contains
                ``robot_id keyframe_id timestamp_ns [ignored...]``, where
                ``robot_id`` is ``ord(char) - ord('a')``.
            robot: Name of the robot whose poses to extract. Matched against the
                decoded character key (e.g. ``'a'``) or, if ``names_override`` is
                provided, against the remapped name.
            frame_id: The frame ID to assign to the returned PathData.
            frame: The coordinate frame convention of this data.
            names_override: Optional dict mapping decoded character keys to
                desired robot names (e.g. ``{'a': 'aerial-07', 'b': 'ground-03'}``).
                Characters not present in the dict are kept as-is.

        Returns:
            PathData instance containing only the poses for the requested robot,
            sorted by keyframe index.
        """

        # Create lookup mapping robot id and keyframe id to timestamp
        time_lookup: Dict[Tuple[int, int], Decimal] = {}
        with open(str(time_path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                robot_id = int(parts[0])
                keyframe_id = int(parts[1])
                timestamp_ns = Decimal(parts[2])
                time_lookup[(robot_id, keyframe_id)] = timestamp_ns / Decimal("1000000000")


        vertices = []
        with open(str(g2o_path), 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not line.startswith("VERTEX_SE3:QUAT"):
                    continue

                parts = line.split()
                key = int(parts[1])
                char = chr(key >> 56)
                idx = key & ((1 << 56) - 1)

                name = names_override.get(char, char) if names_override is not None else char
                if name != robot:
                    continue

                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
                vertices.append((idx, char, [x, y, z], [qx, qy, qz, qw]))

        vertices.sort(key=lambda v: v[0])
        if not vertices:
            raise ValueError(f"No data found for robot '{robot}' in {g2o_path}.")

        timestamps = []
        positions = []
        orientations = []
        for idx, char, pos, ori in vertices:
            robot_id = ord(char) - ord('a')
            timestamps.append(time_lookup[(robot_id, idx)])
            positions.append(pos)
            orientations.append(ori)

        return PathData(
            frame_id=frame_id,
            timestamps=np.array(timestamps, dtype=object),
            positions=np.array(positions, dtype=object),
            orientations=np.array(orientations, dtype=object),
            frame=frame,
        )

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    @staticmethod
    def visualize_2D(dataList: List[PathData], isGTList: List[bool], colorList: List[str], nameList: List[str],
                     save_path: Union[str, None] = None, no_background: bool = False, line_width: float = 1.0,
                     show_grid: bool = False, legend: bool = True,
                     no_border: bool = False, disable_x_label: bool = False, disable_y_label: bool = False,
                     google_maps_scale_bar: bool = False, google_maps_scale_bar_loc: str ="bottom-right",
                     gt_color_lightness_range_val: int = 3,
                     background_image_path: str | None = None,
                     background_image_x_edge: float | None = None, ax: plt.Axes | None = None,
                     background_image_extent_offsets: Union[Tuple[float, float], None] = None,
                     loop_closure_data=None, lc_errors=None, lc_line_width: float = 0.8,
                     title: str | None = None, lc_errors_vmax: float = 50.0):
        """
        Plot all PathData objects on a 2D XY plane.
        
        Args:
            dataList: All PathData objects to plot.
            isGTList: Whether or not each PathData object is GT.
            colorList: Colors to assign to each PathData object (as hex strings starting with #).
            nameList: Robot names corresponding to each PathData object.
            save_path: If provided, figure will be saved to the location. Otherwise, show the figure.
                This does nothing if ax is not None.
            no_background: If true, plot with a transparent background.
            line_width: Width of trajectory lines in the plot.
            show_grid: Whether to draw a grid on the plot.
            legend: If true, include a legend.
            no_border: If true, remove the x & y axes.
            disable_x_label: Remove the x label.
            disable_y_label: Remove the y label.
            google_maps_scale_bar: Whether to add a Google Maps-like scale bar onto the plot.
            google_maps_scale_bar_loc: Location for the scale bar.
            background_image_path: Path to an image to plot in the background; It is assumed
                that the center of the image corresponds to x=0 & y=0 in the PataData frames.
            background_image_x_edge: The distance in meters from center of image to the x edge.
            background_image_extent_offets: XY locations where the image center should be located.
            ax: If passed, plot is drawn onto these axes instead of on a new figure.
        """

        # Check lengths of arguments
        if len(dataList) != len(isGTList) or len(dataList) != len(colorList) or len(dataList) != len(nameList):
            raise ValueError("Lengths of all Lists must be equal!")
        num_data_objs = len(dataList)

        # Check other argument requirements
        if gt_color_lightness_range_val < 0 or gt_color_lightness_range_val >= 20:
            raise ValueError("gt_color_lightness_range_val must be between 0 and 19 inclusive!")

        # Convert hex colors to a palette with varying lightness
        paletteList = []
        for c in colorList:
            # Convert base color to HLS
            rgb = mcolors.to_rgb(c)
            h, _, s = colorsys.rgb_to_hls(*rgb)

            # Generate similar colors with varying lightness
            lightnesses = np.linspace(0.0, 1.0, 20)
            paletteList.append([colorsys.hls_to_rgb(h, li, s) for li in lightnesses])

        # Create the figure or use the provided axes
        created_fig = False
        if ax is None:
            fig, axs = plt.subplots(1, 1)
            created_fig = True
        else:
            axs = ax
            fig = axs.figure
        if no_background:
            fig.patch.set_facecolor('white')
            axs.set_facecolor('none')
        else:
            fig.patch.set_facecolor('white')
            axs.set_facecolor("#F0F0F0")

        # Draw background image
        if background_image_path is not None:
            img = mpimg.imread(background_image_path)
            if background_image_x_edge:
                x_extent_meters = background_image_x_edge
                h, w = img.shape[0], img.shape[1]
                y_extent_meters = x_extent_meters / w * h
                if background_image_extent_offsets is not None:
                    x_offset, y_offset = background_image_extent_offsets
                else:
                    x_offset, y_offset = 0, 0
                extent = [-x_extent_meters + x_offset, x_extent_meters + x_offset,
                          -y_extent_meters + y_offset, y_extent_meters + y_offset]
                axs.imshow(img, extent=extent, origin="upper", alpha=1.0, zorder=0)
            else:
                raise ValueError("Extent must be provided with Background image size via background_image_x_edge.")

        # Calculate trajectory bounds
        all_x = np.concatenate([dec_arr_to_float_arr(path.positions[:,0]) for path in dataList])
        all_y = np.concatenate([dec_arr_to_float_arr(path.positions[:,1]) for path in dataList])
        padding_x = (all_x.max() - all_x.min()) * 0.05
        padding_y = (all_y.max() - all_y.min()) * 0.05
        x_min, x_max = all_x.min() - padding_x, all_x.max() + padding_x
        y_min, y_max = all_y.min() - padding_y, all_y.max() + padding_y

        # Plot loop closures (drawn before trajectories so they appear underneath)
        if loop_closure_data is not None:
            name_to_est: dict = {}
            for _path, _is_gt, _name in zip(dataList, isGTList, nameList):
                if not _is_gt and _name not in name_to_est:
                    name_to_est[_name] = _path

            pos_cache: dict = {}
            for _name, _path in name_to_est.items():
                _ts = dec_arr_to_float_arr(_path.timestamps).astype(float)
                _x = dec_arr_to_float_arr(_path.positions[:, 0]).astype(float)
                _y = dec_arr_to_float_arr(_path.positions[:, 1]).astype(float)
                pos_cache[_name] = (_ts, _x, _y)

            if lc_errors is not None:
                trans_errs = np.asarray(lc_errors["translation_errors"], dtype=float)
                lc_norm = mcolors.Normalize(vmin=0, vmax=lc_errors_vmax, clip=True)
                lc_cmap = mcolors.LinearSegmentedColormap.from_list(
                    "lc_cmap", ["#1a9641", "#a6611a", "#d7191c"])

            for _i in range(loop_closure_data.num_loop_closures):
                _name_a, _name_b = loop_closure_data.names[_i]
                if _name_a not in pos_cache or _name_b not in pos_cache:
                    print("Warning: LC robot not found in pos_cache!")
                    continue
                _ts_a = float(loop_closure_data.timestamps_a[_i])
                _ts_b = float(loop_closure_data.timestamps_b[_i])
                _ts_arr_a, _x_a, _y_a = pos_cache[_name_a]
                _ts_arr_b, _x_b, _y_b = pos_cache[_name_b]
                _xa = float(np.interp(_ts_a, _ts_arr_a, _x_a))
                _ya = float(np.interp(_ts_a, _ts_arr_a, _y_a))
                _xb = float(np.interp(_ts_b, _ts_arr_b, _x_b))
                _yb = float(np.interp(_ts_b, _ts_arr_b, _y_b))
                _color = lc_cmap(lc_norm(trans_errs[_i])) if lc_errors is not None else (1.0, 1.0, 1.0, 0.6)
                axs.plot([_xa, _xb], [_ya, _yb], color=_color, linewidth=lc_line_width, zorder=3)
                axs.plot([_xa, _xb], [_ya, _yb], 'o', color='black', markersize=3, zorder=4)

            if lc_errors is not None:
                _sm = plt.cm.ScalarMappable(norm=lc_norm, cmap=lc_cmap)
                _sm.set_array([])
                fig.colorbar(_sm, ax=axs, label="LC Translation Error (m)", shrink=0.8)

        # Plot the trajectories
        for i in range(num_data_objs):
            label = nameList[i] + (" (GT)" if isGTList[i] else " (Est.)")
            linestyle = ("dotted" if isGTList[i] else None)
            color = (paletteList[i][gt_color_lightness_range_val] if isGTList[i] else paletteList[i][9])
            axs.plot(dataList[i].positions[:,0], dataList[i].positions[:,1],
                     label=label, color=color, linewidth=line_width, linestyle=linestyle, zorder=2)
    
        # Calculate the current aspect ratio and make it match the target
        target_ar = 1.5  
        current_width = x_max - x_min
        current_height = y_max - y_min
        current_ar = current_width / current_height

        if current_ar > target_ar:
            target_height = current_width / target_ar
            diff = target_height - current_height
            y_min -= diff / 2
            y_max += diff / 2
        elif current_ar < target_ar:
            target_width = current_height * target_ar
            diff = target_width - current_width
            x_min -= diff / 2
            x_max += diff / 2

        # Set the new limits that respect the aspect ratio
        axs.set_xlim(x_min, x_max)
        axs.set_ylim(y_min, y_max)
        axs.set_aspect('equal', adjustable='box')
        
        # Make the tick spacing match
        yticks = axs.get_yticks()
        if len(yticks) > 1:
            y_spacing = yticks[1] - yticks[0]
            axs.xaxis.set_major_locator(MultipleLocator(y_spacing))

        # Adjust the spines
        for spine in axs.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.75)

        # Add Grid if Desired
        if show_grid:
            axs.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        else:
            axs.grid(False)

        # Add labels
        if not disable_x_label:
            axs.set_xlabel("X (meters)")
        if not disable_y_label:
            axs.set_ylabel("Y (meters)")

        # Additional configurable parameters
        if legend:
            axs.legend()
        if no_border:
            axs.set_axis_off()

        # Helper function to make a Google Maps like Scale Bar
        def add_google_maps_scale(ax, length_m: float, location: str):
            # Metric calculations
            metric_label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m/1000:.1g} km"
            
            # American calculations
            exact_ft = length_m * 3.28084
            if exact_ft < 5280:
                rounded_ft = int(exact_ft // 25) * 25
                if rounded_ft == 0: 
                    rounded_ft = 25 
                american_label = f"{rounded_ft} ft"
                american_width_m = rounded_ft / 3.28084
            else:
                miles = exact_ft / 5280
                rounded_mi = math.floor(miles * 10) / 10.0
                if rounded_mi == 0: rounded_mi = 0.1
                american_label = f"{rounded_mi:.1f} mi"
                american_width_m = rounded_mi * 1609.34

            # 3. Geometry Setup
            xmin, xmax = ax.get_xlim()
            range_m = xmax - xmin
            
            tick_h = 0.025
            if location == "bottom-right":
                end_x = 0.93
                y_center = 0.12
            elif location == "top-right":
                end_x = 0.93
                y_center = 0.88
            else:
                raise ValueError("location must be 'top-right' or 'bottom-right'")
            
            # Fractional widths for both bars
            frac_m = length_m / range_m
            frac_am = american_width_m / range_m

            # 4. Styling
            main_color = "#373737"
            pe_text = [path_effects.withStroke(linewidth=2.5, foreground='white')]
            pe_line = [path_effects.withStroke(linewidth=5, foreground='white')]
            
            line_params = {'color': main_color, 'transform': ax.transAxes, 'lw': 3.0, 
                        'path_effects': pe_line, 'solid_capstyle': 'round', 'zorder': 10}
            text_params = {'color': main_color, 'transform': ax.transAxes, 'ha': 'right', 
                        'path_effects': pe_text, 'fontsize': 18, 'weight': 'black', 'zorder': 11}

            # --- DRAW METRIC (TOP) ---
            m_start_x = end_x - frac_m
            am_start_x = end_x - frac_am
            # Horizontal and L-brackets (upward)
            ax.plot([m_start_x, end_x], [y_center, y_center], **line_params)
            ax.plot([m_start_x, m_start_x], [y_center, y_center + tick_h], **line_params)
            ax.plot([end_x, end_x], [y_center, y_center + tick_h], **line_params)
            ax.text(am_start_x + 0.9 * (end_x - am_start_x), y_center + 0.01, metric_label, va='bottom', **text_params)

            # --- DRAW AMERICAN (BOTTOM) ---
            # Horizontal and L-brackets (downward)
            ax.plot([am_start_x, end_x], [y_center, y_center], **line_params)
            ax.plot([am_start_x, am_start_x], [y_center, y_center - tick_h], **line_params)
            ax.plot([end_x, end_x], [y_center, y_center - tick_h], **line_params)
            ax.text(am_start_x + 0.9 * (end_x - am_start_x), y_center - 0.018, american_label, va='top', **text_params)

            # Re-draw grey lines on top to clean up the white stroke overlaps
            clean_params = {'color': main_color, 'transform': ax.transAxes, 'lw': 2.5, 'zorder': 12}
            ax.plot([m_start_x, end_x], [y_center, y_center], **clean_params)
            ax.plot([m_start_x, m_start_x], [y_center, y_center + tick_h], **clean_params)
            ax.plot([end_x, end_x], [y_center, y_center + tick_h], **clean_params)
            ax.plot([am_start_x, end_x], [y_center, y_center], **clean_params)
            ax.plot([am_start_x, am_start_x], [y_center, y_center - tick_h], **clean_params)
            ax.plot([end_x, end_x], [y_center, y_center - tick_h], **clean_params)

        # Draw a google maps scale bar
        if google_maps_scale_bar:
            range_m = x_max - x_min
            target_length = range_m * 0.25
            if target_length < 10: suggested_length = target_length
            else: suggested_length = int(round(target_length / 10.0)) * 10
            add_google_maps_scale(axs, suggested_length, google_maps_scale_bar_loc)

        if title is not None:
            axs.set_title(title)

        # Save/Plot the results
        if created_fig:
            if save_path is not None:
                pad = 0.05 if title is not None else 0
                fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=pad)
            else:
                plt.show()
            plt.close(fig)

        return axs

    def visualize_3D(self, otherList: List[PathData], titles: List[str], axes_length: Union[float, List[float]] = 10.0, axes_interval: Union[int, List[int]] = 1000, save_path: Optional[Union[Path, str]] = None):
        """
        Visualizes this PathData (and all others included in otherList) on a single plot.

        Args:
            otherList (List[PathData]): All other PathData objects whose path should also be visualized on this plot.
            titles (List[str]): Titles for each PathData object, starting with self.
            save_path (Path | str | None): If provided, saves the figure to this path instead of displaying it.
        """

        def draw_axes(data: PathData, axes_length: int, axes_interval: int):
            """Helper function that visualizes orientation along the trajectory path with axes."""

            for i in range(0, data.len(), axes_interval):
                # Extract data
                pos = data.positions[i].astype(np.float64)
                quat = data.orientations[i].astype(np.float64)
                rot = R.from_quat(quat)

                # Define unit vectors for X, Y, Z in local frame
                x_axis = rot.apply([1, 0, 0])
                y_axis = rot.apply([0, 1, 0])
                z_axis = rot.apply([0, 0, 1])

                # Plot axes
                ax.quiver(*pos, *x_axis, length=axes_length, color='r', normalize=True, linewidth=0.8)
                ax.quiver(*pos, *y_axis, length=axes_length, color='g', normalize=True, linewidth=0.8)
                ax.quiver(*pos, *z_axis, length=axes_length, color='b', normalize=True, linewidth=0.8)

        # Ensure that the lists are of the proper sizes
        if (len(otherList) + 1) != len(titles):
            raise ValueError("Length of titles should be one more than length of otherlist!")

        # Build a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.positions[:,0].astype(np.float64), 
                self.positions[:,1].astype(np.float64), 
                self.positions[:,2].astype(np.float64), label=titles[0])
        for i, other in enumerate(otherList):
            ax.plot(other.positions[:,0].astype(np.float64), 
                    other.positions[:,1].astype(np.float64), 
                    other.positions[:,2].astype(np.float64), 
                    label=titles[1+i])
            
        # Handle axes_length and axes_interval if they are lists
        if isinstance(axes_length, list):
            if len(axes_length) != (len(otherList) + 1):
                raise ValueError("If axes_length is a list, it must be the same length as otherList + 1!")
        else: axes_length: list[float] = [axes_length] * (len(otherList) + 1)

        if isinstance(axes_interval, list):
            if len(axes_interval) != (len(otherList) + 1):
                raise ValueError("If axes_interval is a list, it must be the same length as otherList + 1!")
        else: axes_interval: list[int] = [axes_interval] * (len(otherList) + 1)

        # Draw orientation axes (X = red, Y = green, Z = blue)
        draw_axes(self, axes_length=axes_length[0], axes_interval=axes_interval[0])
        for i, other in enumerate(otherList):
            draw_axes(other, axes_length=axes_length[i+1], axes_interval=axes_interval[i+1])

        # Set labels
        ax.set_title("Trajectory Comparison with Full Orientation")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        # Concatenate all x, y and z values together
        all_x = self.positions[:,0]
        all_y = self.positions[:,1]
        all_z = self.positions[:,2]
        for other in otherList:
            all_x = np.concatenate((all_x, other.positions[:,0]))
            all_y = np.concatenate((all_y, other.positions[:,1]))
            all_z = np.concatenate((all_z, other.positions[:,2]))
        all_x = all_x.astype(np.float64)
        all_y = all_y.astype(np.float64)
        all_z = all_z.astype(np.float64)

        # Set an equal scale for all axes
        x_center = (all_x.max() + all_x.min()) / 2
        y_center = (all_y.max() + all_y.min()) / 2
        z_center = (all_z.max() + all_z.min()) / 2
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()) / 2
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    
    # =========================================================================
    # ============================ Export Methods =============================
    # =========================================================================

    def to_csv(self, csv_path: Union[Path, str], write_header: bool = True):
        """
        Writes the path data to a .csv file. Note that data will be
        saved in the following order: timestamp, pos.x, pos.y, pos.z,
        ori.w, ori.x, ori.y, ori.z. Timestamp is in seconds.

        Args:
            csv_path (Path | str): Path to the output csv file.
            write_header (bool): If false, skip the header row.

        Raises:
            ValueError: If the output file already exists.
        """

        # setup tqdm
        pbar = tqdm.tqdm(total=None, desc="Saving to csv... ", unit=" frames")

        # Check that file path doesn't already exist
        file_path = Path(csv_path)
        if os.path.exists(file_path):
            raise ValueError(f"Output file already exists: {file_path}")

        # Open csv file
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # Write the first row
            if write_header:
                writer.writerow(['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])

            # Write message data to the csv file
            for i in range(len(self.timestamps)):
                writer.writerow([str(self.timestamps[i]),
                    str(self.positions[i][0]), str(self.positions[i][1]), str(self.positions[i][2]),
                    str(self.orientations[i][3]), str(self.orientations[i][0]), str(self.orientations[i][1]),
                    str(self.orientations[i][2])])
                pbar.update(1)

    def to_txt_file(self, file_path: Union[Path, str], data_to_column: Union[List[int], None] = None):
        """
        Writes the path data to a space-separated text file. This is the inverse
        of :meth:`from_txt`.

        ``data_to_column`` specifies the output column index for each of the 8
        data fields (in this fixed order): timestamp, x, y, z, qw, qx, qy, qz.
        For example, the default ``[0, 1, 2, 3, 4, 5, 6, 7]`` produces::

            timestamp x y z qw qx qy qz

        Set to None to use the default ``[0, 1, 2, 3, 4, 5, 6, 7]``.

        Args:
            file_path (Path | str): Path to the output text file.
            data_to_column (list[int] | None): Maps each data field to its output
                column index. Must be a permutation of ``[0, 1, 2, 3, 4, 5, 6, 7]``.

        Raises:
            ValueError: If the output file already exists.
        """

        # Default column ordering
        if data_to_column is None:
            data_to_column = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            assert np.all(np.array(data_to_column) >= 0)
            assert len(data_to_column) == 8

        # Check that file path doesn't already exist
        out_path = Path(file_path)
        if os.path.exists(out_path):
            raise ValueError(f"Output file already exists: {out_path}")

        # setup tqdm
        pbar = tqdm.tqdm(total=None, desc="Saving to txt... ", unit=" frames")

        with open(out_path, 'w') as f:
            for i in range(len(self.timestamps)):
                row = [''] * 8
                row[data_to_column[0]] = str(self.timestamps[i])
                row[data_to_column[1]] = str(self.positions[i][0])
                row[data_to_column[2]] = str(self.positions[i][1])
                row[data_to_column[3]] = str(self.positions[i][2])
                row[data_to_column[4]] = str(self.orientations[i][3])  # qw
                row[data_to_column[5]] = str(self.orientations[i][0])  # qx
                row[data_to_column[6]] = str(self.orientations[i][1])  # qy
                row[data_to_column[7]] = str(self.orientations[i][2])  # qz
                f.write(' '.join(row) + '\n')
                pbar.update(1)

    def to_tum(self, file_path: Union[Path, str]):
        """
        Writes the path data to a TUM RGB-D dataset trajectory format text file.

        Each row contains 8 space-separated values::

            timestamp x y z q_x q_y q_z q_w

        where ``timestamp`` is in seconds and the orientation quaternion is in
        ``(x, y, z, w)`` order.

        Args:
            file_path (Path | str): Path to the output text file.

        Raises:
            ValueError: If the output file already exists.
        """
        # TUM order: ts x y z qx qy qz qw
        # data fields (ts, x, y, z, qw, qx, qy, qz) → columns (0, 1, 2, 3, 7, 4, 5, 6)
        self.to_txt_file(file_path, data_to_column=[0, 1, 2, 3, 7, 4, 5, 6])

    def to_OdometryData(self, new_frame_id: str, new_child_frame_id: str):
        """ 
        Returns an OdometryData object for this class. 

        Parameters:
            new_frame_id: The new frame ID to assign to the OdometryData object. 
            new_child_frame_id: The new child frame ID to assign to the OdometryData object.
        """

        from .OdometryData import OdometryData
        return OdometryData(frame_id=new_frame_id,
                            child_frame_id=new_child_frame_id,
                            timestamps=self.timestamps,
                            positions=self.positions,
                            orientations=self.orientations,
                            frame=self.frame)

    def to_evo(self) -> PoseTrajectory3D:
        """
        Returns an evo PoseTrajectory3D object for this class.

        Returns:
            PoseTrajectory3D: Trajectory with positions, orientations (wxyz), and timestamps.
        """

        orientations_wxyz = dec_arr_to_float_arr(self.orientations[:, [3, 0, 1, 2]])
        return PoseTrajectory3D(positions_xyz=dec_arr_to_float_arr(self.positions), 
                                orientations_quat_wxyz=orientations_wxyz,
                                timestamps=dec_arr_to_float_arr(self.timestamps))
    
    # =========================================================================
    # ======================= Multi PathData Methods ========================== 
    # ========================================================================= 

    @staticmethod
    def make_start_and_end_times_match(est: list[PathData], gt: list[PathData]) -> Tuple[list[PathData], list[PathData]]:
        """ 
        For pairs of lists of PathData objects, extract each pair by index and 
        ensure that the first and last timestamps match by extending the data
        as necessary at the start and end with duplicate values. Used for evaluation
        purposes.
        
        Mimics the behavior found in ROMAN's (https://github.com/lunarlab-gatech/roman) evaluation scripts.

        Parameters:
            est: List of PathData objects that represent estimated paths.
            gt: List of PathData objects that represent ground truth paths.
        """

        # Copy the PathData objects so we don't modify the originals
        est = copy.deepcopy(est)
        gt = copy.deepcopy(gt)

        # Check that the lists are the same length
        if len(est) == 0 or len(gt) == 0 or len(est) != len(gt):
            raise ValueError("est and gt lists must be non-empty and of the same length!")

        # For each pair of PathData objects
        for est_i, gt_i in zip(est, gt):

            # Adjust start times
            if est_i.timestamps[0] < gt_i.timestamps[0]:
                gt_i.timestamps = np.concatenate(([est_i.timestamps[0]], gt_i.timestamps))
                gt_i.positions = np.concatenate(([gt_i.positions[0]], gt_i.positions))
                gt_i.orientations = np.concatenate(([gt_i.orientations[0]], gt_i.orientations))
            elif est_i.timestamps[0] > gt_i.timestamps[0]:
                est_i.timestamps = np.concatenate(([gt_i.timestamps[0]], est_i.timestamps))
                est_i.positions = np.concatenate(([est_i.positions[0]], est_i.positions))
                est_i.orientations = np.concatenate(([est_i.orientations[0]], est_i.orientations))

            # Adjust end times
            if est_i.timestamps[-1] < gt_i.timestamps[-1]:
                est_i.timestamps = np.concatenate((est_i.timestamps, [gt_i.timestamps[-1]]))
                est_i.positions = np.concatenate((est_i.positions, [est_i.positions[-1]]))
                est_i.orientations = np.concatenate((est_i.orientations, [est_i.orientations[-1]]))
            elif est_i.timestamps[-1] > gt_i.timestamps[-1]:
                gt_i.timestamps = np.concatenate((gt_i.timestamps, [est_i.timestamps[-1]]))
                gt_i.positions = np.concatenate((gt_i.positions, [gt_i.positions[-1]]))
                gt_i.orientations = np.concatenate((gt_i.orientations, [gt_i.orientations[-1]]))
        
        # Return the modified lists
        return est, gt

    @staticmethod
    def concatenate_PathData(path_data_objs: list[PathData]) -> PathData:
        """
        Combines multiple PathData objects into a single PathData object. In doing so,
        will shift the timestamps of each subsequent PathData so that their data starts
        one second after the previous PathData ends. Also assumes the frame_id and frame
        of the first PathData object for final PathData object.

        Mimics the behavior found in ROMAN's (https://github.com/lunarlab-gatech/roman) evaluation scripts.

        Args:
            path_data_objs: List of PathData objects to concatenate.

        Returns:
            PathData: A single PathData with all trajectories joined end-to-end.

        Raises:
            ValueError: If the list is empty or has only one element.
        """

        print("Warning! This code has not been unit tested yet!")

        if len(path_data_objs) == 0:
            raise ValueError("path_data_objs list is empty!")
        if len(path_data_objs) == 1:
            raise ValueError("path_data_objs list has only one element; no need to concatenate!")

        # NOTE: Assumes the frame_id and frame of the first object
        frame_id = path_data_objs[0].frame_id
        frame = path_data_objs[0].frame

        # Create all empty arrays to hold concatenated data
        all_timestamps = np.zeros((0,), dtype=Decimal)
        all_positions = np.zeros((0, 3), dtype=Decimal)
        all_orientations = np.zeros((0, 4), dtype=Decimal)

        # For each PathData object
        for i, path_data in enumerate(path_data_objs):

            # If not first PathData, shift timestamps
            if i == 0:
                shifted_timestamps = path_data.timestamps
            else:
                shifted_timestamps = path_data.timestamps - path_data.timestamps[0] + all_timestamps[-1] + 1

            # Concatentate data
            all_timestamps = np.concatenate((all_timestamps, shifted_timestamps), axis=0)
            all_positions = np.concatenate((all_positions, path_data.positions), axis=0)
            all_orientations = np.concatenate((all_orientations, path_data.orientations), axis=0)

        return PathData(frame_id, all_timestamps, all_positions, all_orientations, frame)
    
    @staticmethod
    def seperate_PathData(original_PathDatas: list[PathData], merged_PathData: PathData) -> list[PathData]:
        """
        Inverse of concatenate_PathData(); needs the original objects to know the timestamps
        that each trajectory is found on.

        Args:
            original_PathDatas: The original PathData objects used during concatenation,
                needed to determine timestamp boundaries.
            merged_PathData: The single merged PathData to split apart.

        Returns:
            list[PathData]: One PathData per original trajectory, cropped from the merged data.
        """

        # Calculate the start and end times for the beginning of each robots data
        start_times: list[float] = []
        end_times: list[float] = []
        for i in range(len(original_PathDatas)): 
            pd: PathData = original_PathDatas[i]
            if i == 0:
                start_times.append(pd.timestamps[0])
                end_times.append(pd.timestamps[-1])
            else:
                start_times.append(end_times[-1] + 1)
                end_times.append(pd.timestamps[-1] - pd.timestamps[0] + end_times[-1] + 1)

        # Make deep copies of trajectories
        merged_PathData_copies: list[PathData] = [copy.deepcopy(merged_PathData) for _ in range(len(original_PathDatas))]

        # Reduce trajectories to the specific time that covers each robot, then restore original timestamps
        for i, pd in enumerate(merged_PathData_copies):
            pd.crop_data(Decimal(start_times[i]), Decimal(end_times[i]))
            if i > 0:
                offset = Decimal(start_times[i]) - original_PathDatas[i].timestamps[0]
                pd.timestamps = pd.timestamps - offset
        return merged_PathData_copies

    @staticmethod
    def align_and_calculate_traj_errors(gt_path: PathData, est_path: PathData, max_diff: float, visualize: bool = False, 
            axes_length: Union[float, list[float]] = 10.0, axes_interval: Union[int, list[int]] = 1000) -> Tuple[dict, PathData, PathData]:
        """
        Utilizing the evo library, calculates a variety of trajectory error metrics
        and returns them in a dictionary. Also returns aligned PathData objects.

        Parameters:
            max_diff: maximum absolute time difference allowed between associated timestamps
            visualize: If true, will show a 3D plot of the aligned trajectories.
            axes_length: Same as in visualize() method.
            axes_interval: Same as in visualize() method.
        """

        gt_traj: PoseTrajectory3D = gt_path.to_evo()
        est_traj: PoseTrajectory3D = est_path.to_evo()

        gt_traj, est_traj = sync.associate_trajectories(gt_traj, est_traj, max_diff)


        est_traj_align: PoseTrajectory3D = copy.deepcopy(est_traj)
        est_traj_align.align(gt_traj, correct_scale=False, correct_only_scale=False) 

        path_pair: Tuple[PoseTrajectory3D, PoseTrajectory3D] = (gt_traj, est_traj_align)

        # Calculate various error metrics using evo, including APE and RPE
        all_pose_relations: List[metrics.PoseRelation] = [metrics.PoseRelation.full_transformation, # dimensionless
                                                          metrics.PoseRelation.translation_part, # meters
                                                          metrics.PoseRelation.rotation_part, # dimensionless
                                                          metrics.PoseRelation.rotation_angle_deg, # degrees
                                                          metrics.PoseRelation.rotation_angle_rad, # radians
                                                          metrics.PoseRelation.point_distance, # meters
                                                          metrics.PoseRelation.point_distance_error_ratio] # percent
        all_statistic_types: List[metrics.StatisticsType] = [metrics.StatisticsType.rmse,
                                                             metrics.StatisticsType.mean,
                                                             metrics.StatisticsType.median,
                                                             metrics.StatisticsType.std,
                                                             metrics.StatisticsType.min,
                                                             metrics.StatisticsType.max,
                                                             metrics.StatisticsType.sse]
        all_metrics: List = [metrics.APE, metrics.RPE]
        dict_all_results: dict = {}
        for metric in all_metrics:
            dict_metric: dict = {}

            for pose_relation in all_pose_relations:
                dict_relation: dict = {}

                # Skip uncompatible relation with metric
                if metric is metrics.APE and pose_relation == metrics.PoseRelation.point_distance_error_ratio:
                    continue

                path_pair_copied = copy.deepcopy(path_pair)
                metric_with_relation: metrics.PE = metric(pose_relation)
                metric_with_relation.process_data(path_pair_copied)

                for stat in all_statistic_types:
                    final_stat: float = metric_with_relation.get_statistic(stat)
                    dict_relation[stat.name] = final_stat

                dict_metric[pose_relation.name] = dict_relation
            
            dict_all_results[metric.__name__] = dict_metric
            
        # Convert the aligned trajectory data back to PathData objects
        est_traj_align_pathdata = PathData.from_evo(est_traj_align, est_path.frame_id, est_path.frame)
        gt_traj_pathdata = PathData.from_evo(gt_traj, gt_path.frame_id, gt_path.frame)

        # Visualize the aligned trajectories if desired
        if visualize:
            gt_traj_pathdata.visualize_3D([est_traj_align_pathdata], ['Ground Truth', 'Estimated (Aligned)'], 
                              axes_interval=axes_interval, axes_length=axes_length)
        
        return dict_all_results, est_traj_align_pathdata, gt_traj_pathdata
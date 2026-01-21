from __future__ import annotations

import decimal
from decimal import Decimal
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from rosbags.typesys import Stores, get_typestore
import struct
import sys
from pathlib import Path
import tqdm
from typeguard import typechecked
from typing import Union, List, Tuple, Optional, Any, Callable

from ..conversion_utils import col_to_dec_arr
from ..ModuleImporter import ModuleImporter
from .Data import Data, CoordinateFrame, ROSMsgLibType

@typechecked
class LiDARData(Data):
    """
    LiDAR Data class that contains LiDAR-specific attributes and methods.
    Inherits from the generic Data class.

    NOTE: Assumes points at [0,0,0] are invalid and will set them to NaNs when necessary. 
    Thus, all other points are assumed to be valid.
    """

    point_clouds: List[np.ndarray] # List of length T of (N, 3) arrays of point clouds, with assumed (x, y, z) ordering
    channels: Optional[NDArray] # List of length T of (N) arrays with channel number for each point
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: np.ndarray | list, point_clouds: List[np.ndarray], 
                 channels: Optional[NDArray], frame: CoordinateFrame) -> None:
        super().__init__(frame_id, timestamps)
        self.point_clouds = point_clouds
        self.channels = channels
        self.frame = frame

        # Used to transform LiDAR data
        self.transformations: List[Callable] = []

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    def from_npy_files(cls, npy_folder_path: Union[Path, str], frame_id: str, frame: CoordinateFrame) -> LiDARData:
        """
        Load LiDAR data from a series of .npy files in a specified folder,
        where file names correspond to timestamps. Each point is expected to
        contain either [x, y, z] or [x, y, z, channel].

        Args:
            npy_folder_path: Path to the folder.
            frame_id: The frame ID for the LiDAR data.
            frame: The coordinate frame of the LiDAR data.
        Returns:
            LiDARData: An instance of LiDARData populated with the loaded data.
        """

        # Get all npy files in the designated folder (sorted)
        all_npy_files: List[str] = [str(p) for p in Path(npy_folder_path).glob("*.npy")]
        print(f"Found {len(all_npy_files)} .npy files in folder {npy_folder_path}")

        # Extract the timestamps and sort the npy files
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_npy_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]
        all_npy_files_sorted = [all_npy_files[i] for i in sorted_indices]

        # Check shape
        first_pc = np.load(all_npy_files_sorted[0], mmap_mode='r')
        assert len(first_pc.shape) == 2
        assert first_pc.shape[1] in (3, 4)
        has_channels = first_pc.shape[1] == 4

        # Setup arrays to hold data
        point_clouds_memmap: List[np.ndarray] = []
        channels_memmap: Optional[List[np.ndarray]] = [] if has_channels else None
            
        # Load all the point clouds memory mapped (so they are kept on disk)
        pbar = tqdm.tqdm(total=len(all_npy_files_sorted), desc="Extracting Point Clouds...", unit=" files")
        for i, path in enumerate(all_npy_files_sorted):
            pc = np.load(path, mmap_mode="r")
            assert pc.shape == first_pc.shape

            # Separate XYZ and optional channel
            point_clouds_memmap.append(pc[:, :3])
            if has_channels:
                channels_memmap.append(pc[:, 3])
            pbar.update()

        # Return an LiDARData class
        return cls(frame_id, timestamps_sorted, point_clouds_memmap, channels_memmap, frame)
    
    # def to_npy_files(self, output_folder_path: Union[Path, str]) -> None:
    #     """
    #     Save LiDAR point clouds to a series of .npy files in a specified folder,
    #     where file names correspond to timestamps.

    #     Args:
    #         output_folder_path: Path to the output folder.
    #     """

    #     # Create the output folder
    #     output_folder_path = Path(output_folder_path)
    #     output_folder_path.mkdir(parents=True, exist_ok=True)

    #     # Save each .npy file
    #     num_frames = self.point_clouds.shape[0]
    #     pbar = tqdm.tqdm(total=num_frames, desc="Saving Point Clouds...", unit=" files")
    #     for i in range(num_frames):
    #         if self.channels is not None:
    #             pc = np.concatenate([self.point_clouds[i], self.channels[i][:, None]], axis=1)
    #         else:
    #             pc = self.point_clouds[i]
    #         np.save(output_folder_path / f"{self.timestamps[i]}.npy", pc)
    #         pbar.update()
    #     pbar.close()

    # =========================================================================
    # ========================= Reproducible Loading ========================== 
    # =========================================================================  
    def get_point_cloud_at_index(self, index: int):
        """ 
        Gets the point cloud at index T and ensures all necessary transformations are applied.
        This is a safe copy of the memmapped array, meaning it can be transformed and changed. 
        """

        pc = self.point_clouds[index].astype(np.float32, copy=True) 
        channels = None
        if self.channels is not None:
            channels = self.channels[index].astype(np.int16, copy=True)

        # Mask invalid points (all zeros) and set to NaNs
        mask_invalid = (pc == 0.0).all(axis=1) | np.isnan(pc).all(axis=1)
        pc[mask_invalid] = np.nan

        # Apply any other requested transformations
        for trans in self.transformations:
            pc, channels = trans(pc, channels)

        return pc, channels
    
    # =========================================================================
    # ========================= Manipulation Methods ========================== 
    # =========================================================================  
    def calculate_point_channels(self, num_channels: int, v_min_angle: float, v_max_angle: float) -> None:
        """ 
        Calculate channel numbers for each point.
         
        NOTE: This assumes that lasers are evenly spaced within the angular range and that
        the first laser fires at v_min_angle and the last laser fires at v_max_angle.
        NOTE: Invalid points (NaNs) get a channel of -1.
        """

        if self.channels is not None:
            raise RuntimeError("Attempted to calculate channel numbers, but its already calculated!")

        # Calculate laser line angles
        laser_angles = np.linspace(v_min_angle, v_max_angle, num_channels)

        # Compute channels
        channels = []
        pbar = tqdm.tqdm(total=self.len(), desc="Calculating Point Channels...", unit=" frames")
        for i in range(self.len()):
            # Get point cloud
            pc, _ = self.get_point_cloud_at_index(i)

            # Extract dimensions
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]

            # Compute vertical angle per point in degrees
            horiz_dist = np.sqrt(x**2 + y**2)
            vertical_angle = np.arctan2(z, horiz_dist) * 180.0 / np.pi

            # Assign points to laser line that is closest to its angle
            angle_diff = np.abs(vertical_angle[..., None] - laser_angles)
            chan = np.argmin(angle_diff, axis=-1)

            # Any point where x, y, or z is NaN gets a channel of -1
            mask_invalid = np.isnan(pc).any(axis=1)
            chan[mask_invalid] = -1
            channels.append(chan.astype(np.int16))
            pbar.update()

        self.channels = channels
        pbar.close()

    def make_dense(self):
        """ Removes invalid points (infinity and NaNs) to make the point cloud dense. """

        def dense_transformation(pts: np.ndarray, channels: Optional[np.ndarray]) -> np.ndarray:
            """
            Args:
                pts: A (N, 3) point cloud array.
                channels: A (N) channel array.
            Returns:
                A filtered (M, 3) point cloud with only valid points.
            """

            valid_mask = np.isfinite(pts).all(axis=1)
            return pts[valid_mask], channels[valid_mask]


        if dense_transformation not in self.transformations:
            self.transformations.append(dense_transformation)

    # =========================================================================
    # ============================ Visualization ============================== 
    # =========================================================================  

    @typechecked
    def visualize(self, interval_ms: int = 100, plot_lims: Tuple[float, float] = (-50.0, 50.0), testing: bool = False):
        """
        Visualizes the raw LiDAR data over time using Matplotlib FuncAnimation.

        Parameters:
            interval_ms: The time between plotted frames.
            plot_lims: The axes limits of the 3D plot.
            testing: Only used for test cases, disables blocking in plt.show()
        """

        # Create the plot
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(*plot_lims)
        ax.set_ylim(*plot_lims)
        ax.set_zlim(*plot_lims)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)
        ax.set_zlabel("Z", fontsize=20)
        title = ax.set_title(f"", fontsize=24)

        # Create the update function
        scatter = ax.scatter([], [], [], s=4, cmap='viridis')
        def update(frame: int):
            # Load safe copy of memmap array for plotting
            pts, channels = self.get_point_cloud_at_index(frame)

            # Mask invalid points
            nan_mask = np.isnan(pts).any(axis=1)
            pts = pts[~nan_mask]
            channels = channels[~nan_mask]

            # Update the plot
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            scatter._offsets3d = (x, y, z)  # update Z
            if channels is not None:
                scatter.set_array(channels)
            else:
                scatter.set_array(z)  # Set color based on Z
            title.set_text(f"LiDAR Frame {frame+1}/{self.len()-1}")
            return scatter, title
    
        # Start the animation
        ani = FuncAnimation(fig, update, frames=self.len(), interval=interval_ms, blit=False, repeat=False)
        if not testing:
            plt.show()

    # =========================================================================
    # =================== ROS Message Conversion Methods =====================
    # =========================================================================

    @staticmethod 
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """ Return the __msgtype__ for a PointCloud2 msg. """

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            return typestore.types['sensor_msgs/msg/PointCloud2'].__msgtype__
        elif lib_type == ROSMsgLibType.RCLPY:
            return ModuleImporter.get_module_attribute('sensor_msgs.msg', 'PointCloud2')
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for LiDARData.get_ros_msg_type()!")

    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int):
        """
        Gets a PointCloud2 message corresponding to the point cloud at index i.

        Args:
            lib_type (ROSMsgLibType): The type of ROS message to return (e.g., ROSBAGS, RCLPY).
            i (int): The index of the point cloud to convert.
        Raises:
            ValueError: If i is outside the data bounds.

        NOTE: Currently publishes an unordered point cloud.
        NOTE: Assumes all points are collected at same time (likely false in the real-world).
        NOTE: Assumes channels data is provided.
        NOTE: Assumes intensity of 255 for all points.
        """
        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = int((self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9"))

        # Get point cloud with NaN masking applied
        pts, channels = self.get_point_cloud_at_index(i)
        num_points = pts.shape[0]

        # Calculate time and intensity (NOTE: Time is assumed to be zero & intensity assumed to be 255 for all points)
        time = np.zeros((num_points, 1), dtype=np.float32)
        intensity = np.ones((num_points, 1), dtype=np.float32) * 255

        # Append channel and time onto our point cloud to get (N, 6)
        if channels is not None:
            pc_aug = np.concatenate([pts, channels[:, np.newaxis], time, intensity], axis=-1)
        else:
            raise RuntimeError("ROS2 PointCloud2 message expects channels data, but it has not been provided or calculated via calculate_point_channels()!")

        # Create PointCloud2 message
        if lib_type == ROSMsgLibType.ROSBAGS:

            typestore = get_typestore(Stores.ROS2_HUMBLE)
            PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
            Header = typestore.types['std_msgs/msg/Header']
            Time = typestore.types['builtin_interfaces/msg/Time']
            PointField = typestore.types['sensor_msgs/msg/PointField']

            # Define point fields for x, y, z
            fields = [
                PointField(name='x', offset=0, datatype=7, count=1),   # FLOAT32 = 7
                PointField(name='y', offset=4, datatype=7, count=1),
                PointField(name='z', offset=8, datatype=7, count=1),
                PointField(name="ring", offset=12, datatype=7, count=1),
                PointField(name="time", offset=16, datatype=7, count=1),
                PointField(name="intensity", offset=20, datatype=7, count=1)
            ]

            return PointCloud2(
                header=Header(
                    stamp=Time(sec=int(seconds), nanosec=int(nanoseconds)),
                    frame_id=self.frame_id
                ),
                height=1,
                width=num_points,
                fields=fields,
                is_bigendian= (sys.byteorder == "big"),
                point_step=24,  # 3 floats * 4 bytes
                row_step=24 * num_points,
                data=np.frombuffer(pc_aug.astype(np.float32).tobytes(), dtype=np.uint8),
                is_dense=not np.isnan(pts).any()
            )

        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:

            Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')
            PointCloud2 = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'PointCloud2')
            PointField = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'PointField')

            # Create the message object
            pc_msg = PointCloud2()
            pc_msg.header = Header()
            if lib_type == ROSMsgLibType.RCLPY: 
                Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                pc_msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
            else:
                rospy = ModuleImporter.get_module('rospy')
                pc_msg.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))
            pc_msg.header.frame_id = self.frame_id

            # Set the height and width assuming an unordered point cloud
            pc_msg.height = 1
            pc_msg.width = num_points

            # Set the point fields
            pc_msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="ring", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=20, datatype=PointField.FLOAT32, count=1)
            ]
            pc_msg.point_step = 24
            pc_msg.row_step = pc_msg.point_step * num_points

            # Fill in the remaining data
            pc_msg.is_bigendian = sys.byteorder == "big"
            pc_msg.is_dense = not np.isnan(pts).any()

            # Pack points into binary
            fmt = "<ffffff" if not pc_msg.is_bigendian else ">ffffff"
            pc_msg.data = b"".join(struct.pack(fmt, *p) for p in pc_aug)
            return pc_msg

        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for LiDARData.get_ros_msg()!")
        
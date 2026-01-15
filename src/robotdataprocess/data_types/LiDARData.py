from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from .Data import Data, CoordinateFrame, ROSMsgLibType
import decimal
from decimal import Decimal
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from ..ModuleImporter import ModuleImporter
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import struct
import sys
from typeguard import typechecked
from typing import Union, List, Tuple, Optional, Any
import tqdm

@typechecked
class LiDARData(Data):
    """
    LiDAR Data class that contains LiDAR-specific attributes and methods.
    Inherits from the generic Data class.

    NOTE: Assumes points at [0,0,0] are invalid and sets them to NaNs. 
    Thus, all other points are assumed to be valid.
    """

    point_clouds: NDArray # (T, N, 3) array of point clouds, with assumed (x, y, z) ordering
    channels: Optional[NDArray] # (T, N) array with channel number for each point
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: np.ndarray | list, point_clouds: NDArray, channels: Optional[NDArray], frame: CoordinateFrame) -> None:
        super().__init__(frame_id, timestamps)
        self.point_clouds = point_clouds
        self.channels = channels
        self.frame = frame

        # Set points at the origin to NaNs
        mask = (point_clouds == 0).all(axis=-1) 
        point_clouds[mask] = np.nan

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    def from_npy_files(cls, npy_folder_path: Union[Path, str], frame_id: str, frame: CoordinateFrame) -> LiDARData:
        """
        Load LiDAR data from a series of .npy files in a specified folder,
        where file names correspond to timestamps.

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

        # Extract the timestamps and sort them
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_npy_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]

        # Use sorted_indices to sort all_image_files in the same way
        all_npy_files_sorted = [all_npy_files[i] for i in sorted_indices]

        # Check the point cloud shape from the first file
        first_pc = np.load(all_npy_files_sorted[0], 'r')
        assert len(first_pc.shape) == 2
        assert first_pc.shape[1] == 3

        # Load all the point clouds into a single array
        point_clouds = np.zeros((len(all_npy_files_sorted), first_pc.shape[0], 3), dtype=np.float64)
        pbar = tqdm.tqdm(total=len(all_npy_files_sorted), desc="Extracting Point Clouds...", unit=" files")
        for i, path in enumerate(all_npy_files_sorted):
            point_clouds[i] = np.load(path, 'r')
            assert point_clouds[i].shape[0] == first_pc.shape[0]
            pbar.update()

        # Return an LiDARData class
        return cls(frame_id, timestamps_sorted, point_clouds, None, frame)
    
    # =========================================================================
    # ========================= Manipulation Methods ========================== 
    # =========================================================================  

    def calculate_point_channels(self, num_channels: int, v_min_angle: float, v_max_angle: float) -> None:
        """ Calculate channel numbers for each point """

        if self.channels is not None:
            raise RuntimeError("Attempted to calculate channel numbers, but its already calculated!")

        x = self.point_clouds[..., 0]
        y = self.point_clouds[..., 1]
        z = self.point_clouds[..., 2]

        # Compute vertical angle in degrees
        horiz_dist = np.sqrt(x**2 + y**2)
        vertical_angle = np.arctan2(z, horiz_dist) * 180.0 / np.pi

        # Map angle to channel index
        angle_range = v_max_angle - v_min_angle
        channels = (vertical_angle - v_min_angle) / angle_range * (num_channels - 1)

        # Round and clip to valid channel numbers
        channels = np.round(channels).astype(int)
        channels = np.clip(channels, 0, num_channels - 1)
        self.channels = channels

    # =========================================================================
    # =========================== Frame Conversions =========================== 
    # ========================================================================= 
    def to_FLU_frame(self):
        # If we are already in the FLU frame, return
        if self.frame == CoordinateFrame.FLU:
            print("Data already in FLU coordinate frame, returning...")
            return

        # If in NED, run the conversion
        elif self.frame == CoordinateFrame.NED:
            R_NED = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]])
            for i in range(self.point_clouds.shape[0]):
                self.point_clouds[i] = (R_NED @ self.point_clouds[i].T).T
            self.frame = CoordinateFrame.FLU

        else:
            raise RuntimeError(f"LiDARData class is in an unexpected frame: {self.frame}!")

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
            # Get only valid points
            pts = self.point_clouds[frame].astype(float)
            valid_pts_mask = ~np.isnan(pts).any(axis=1)
            pts = pts[valid_pts_mask]
            channels = self.channels[frame][valid_pts_mask]

            # Update the plot
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            scatter.set_offsets(np.c_[x, y])  # update X and Y
            scatter.set_3d_properties(z, zdir='z')  # update Z
            if self.channels is not None:
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
    # =========================== Conversion to ROS =========================== 
    # ========================================================================= 

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """ Return the __msgtype__ for an LiDAR (Point Cloud) msg. """

        if lib_type == ROSMsgLibType.RCLPY:
            return ModuleImporter.get_module_attribute('sensor_msgs.msg', 'PointCloud2')
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for OdometryData.get_ros_msg_type()!")
            
    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int):
        """
        Gets an Image ROS2 message corresponding to the LiDAR scan.
        
        Args:
            lib_type: The ROS library we're getting the message for.
            i (int): The index of the image message to convert.
        Raises:
            ValueError: If i is outside the data bounds.

        NOTE: Currently uses an unordered point cloud.
        NOTE: Does not publish intensity, and assumes all points collected at same time (only holds true for simulation)
        NOTE: Assumes channels data is provided.
        NOTE: Assumes intensity of 255 for all points.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = (self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) \
                       * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN)

        # Write the data into the new msg
        if lib_type == ROSMsgLibType.RCLPY:
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
            num_points = self.point_clouds[i].shape[0]
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
            pc_msg.is_bigendian = True if sys.byteorder == "big" else False
            pc_msg.is_dense = not np.isnan(self.point_clouds[i]).any()

            # Calculate time and intensity (NOTE: Time is assumed to be zero & intensity assumed to be 255 for all points)
            time = np.zeros((num_points, 1), dtype=np.float32)
            intensity = np.ones((num_points, 1), dtype=np.float32) * 255

            # Append channel and time onto our point cloud to get (T, N, 5)
            if self.channels is not None:
                pc_aug = np.concatenate([self.point_clouds[i], self.channels[i][:, np.newaxis], time, intensity], axis=-1)
            else:
                raise RuntimeError("ROS2 PointCloud2 message expects channels data, but it has not been provided or calculated via calculate_point_channels()!")

            # Pack points into binary
            fmt = "<ffffff" if not pc_msg.is_bigendian else ">ffffff"
            pc_msg.data = b"".join(struct.pack(fmt, *p) for p in pc_aug)
            return pc_msg

        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for OdometryData.get_ros_msg()!")
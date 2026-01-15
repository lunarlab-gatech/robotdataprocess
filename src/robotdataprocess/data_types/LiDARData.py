from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from .Data import Data, CoordinateFrame
from decimal import Decimal
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from .Data import Data, ROSMsgLibType
from decimal import Decimal
import decimal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from rosbags.typesys import Stores, get_typestore
import struct
from typeguard import typechecked
from typing import Union, List, Tuple
from typing import Union, List, Any
import tqdm

@typechecked
class LiDARData(Data):
    """
    LiDAR Data class that contains LiDAR-specific attributes and methods.
    Inherits from the generic Data class.
    """

    point_clouds: NDArray[Decimal] # (T, N, 3) array of point clouds, with assumed (x, y, z) ordering
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: np.ndarray | list, point_clouds: NDArray, frame: CoordinateFrame) -> None:
        super().__init__(frame_id, timestamps)
        self.point_clouds = point_clouds
        self.frame = frame

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
        first_pc = np.load(all_npy_files_sorted[0], mmap_mode='r')
        # check shape
        assert len(first_pc.shape) == 2
        assert first_pc.shape[1] == 3

        # Load all the point clouds into a single array
        point_clouds = np.zeros((len(all_npy_files_sorted), first_pc.shape[0], 3), dtype=np.float64)
        pbar = tqdm.tqdm(total=len(all_npy_files_sorted), desc="Extracting Point Clouds...", unit=" files")
        for i, path in enumerate(all_npy_files_sorted):
            point_clouds[i] = np.load(path, mmap_mode='r')
            assert point_clouds[i].shape[0] == first_pc.shape[0]
            pbar.update()

        # Return an LiDARData class
        return cls(frame_id, timestamps_sorted, point_clouds, frame)

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
            pts = self.point_clouds[frame].astype(float)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            scatter.set_offsets(np.c_[x, y])  # update X and Y
            scatter.set_3d_properties(z, zdir='z')  # update Z
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
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            from sensor_msgs.msg import PointCloud2
            return PointCloud2
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for LiDARData.get_ros_msg_type()!")

    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int):
        """
        Gets a PointCloud2 ROS2 Humble message corresponding to the point cloud at index i.

        Args:
            lib_type (ROSMsgLibType): The type of ROS message to return (e.g., ROSBAGS, RCLPY).
            i (int): The index of the point cloud to convert.
        Raises:
            ValueError: If i is outside the data bounds.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = int((self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN))

        # Get the point cloud data (N, 3) array
        points = self.point_clouds[i]
        num_points = points.shape[0]

        # Convert points to binary data (x, y, z as float32)
        point_data = points.astype(np.float32).tobytes()

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
                PointField(name='z', offset=8, datatype=7, count=1)
            ]

            return PointCloud2(
                header=Header(
                    stamp=Time(sec=int(seconds), nanosec=int(nanoseconds)),
                    frame_id=self.frame_id
                ),
                height=1,
                width=num_points,
                fields=fields,
                is_bigendian=False,
                point_step=12,  # 3 floats * 4 bytes
                row_step=12 * num_points,
                data=np.frombuffer(point_data, dtype=np.uint8),
                is_dense=True
            )

        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            from std_msgs.msg import Header
            from sensor_msgs.msg import PointCloud2, PointField

            # Create the message objects
            pc2_msg = PointCloud2()
            pc2_msg.header = Header()

            if lib_type == ROSMsgLibType.RCLPY:
                from rclpy.time import Time
                pc2_msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
            else:  # ROSPY
                import rospy
                pc2_msg.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))

            pc2_msg.header.frame_id = self.frame_id
            pc2_msg.height = 1
            pc2_msg.width = num_points

            # Define point fields
            pc2_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]

            pc2_msg.is_bigendian = False
            pc2_msg.point_step = 12
            pc2_msg.row_step = 12 * num_points
            pc2_msg.data = point_data
            pc2_msg.is_dense = True

            return pc2_msg

        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for LiDARData.get_ros_msg()!")

from ..utils.conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from .Data import CoordinateFrame, ROSMsgLibType
from .SequentialData import SequentialData
import decimal
from decimal import Decimal
from ..utils.ModuleImporter import ModuleImporter
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from robotdataprocess.data_types.PathData import PathData
from ..ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag1 import Reader as Reader1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
from typing import Union, Any, Optional, List
import tqdm

@typechecked
class ImuData(SequentialData):
    """
    IMU sensor data with linear acceleration, angular velocity, and optional orientation.

    Supports loading from ROS2 bags and TXT files (TartanAir format). Can be
    integrated via Euler's method to produce a PathData trajectory, or published
    as ``sensor_msgs/Imu`` ROS messages.

    Attributes:
        lin_acc: (N, 3) array of linear accelerations in m/s^2.
        ang_vel: (N, 3) array of angular velocities in rad/s.
        orientations: (N, 4) array of quaternions (x, y, z, w), or None if unavailable.
        frame: The coordinate frame convention of this data.
    """

    # Define IMU-specific data attributes
    lin_acc: NDArray
    ang_vel: NDArray
    orientations: Optional[NDArray] # quaternions (x, y, z, w)
    frame: CoordinateFrame

    @typechecked
    def __init__(self, frame_id: str, frame: CoordinateFrame, timestamps: Union[np.ndarray, list], 
                 lin_acc: Union[np.ndarray, list], ang_vel: Union[np.ndarray, list],
                 orientations: Optional[NDArray]):
        
        # Copy initial values into attributes
        super().__init__(frame_id, timestamps)
        self.frame = frame
        self.lin_acc = col_to_dec_arr(lin_acc)
        self.ang_vel = col_to_dec_arr(ang_vel)
        if orientations is not None:
            self.orientations = col_to_dec_arr(orientations)
        else:
            self.orientations = None

        # Check to ensure that all arrays have same length
        if len(self.timestamps) != len(self.lin_acc) or len(self.lin_acc) != len(self.ang_vel) \
            or (self.orientations is not None and len(self.ang_vel) != len(self.orientations)):
            raise ValueError("Lengths of timestamp, lin_acc, ang_vel, and orientation arrays are not equal!")

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in ImuData. """
        pass

    def __eq__(self, other) -> bool:
        parent_result = super().__eq__(other)
        if parent_result is not True:
            return parent_result
        if self.frame != other.frame:
            print(f"  [__eq__] frame: {self.frame} != {other.frame}")
            return False
        if not np.array_equal(self.lin_acc, other.lin_acc):
            print(f"  [__eq__] lin_acc not equal")
            return False
        if not np.array_equal(self.ang_vel, other.ang_vel):
            print(f"  [__eq__] ang_vel not equal")
            return False
        if (self.orientations is None) != (other.orientations is None):
            print(f"  [__eq__] orientations: {self.orientations} != {other.orientations}")
            return False
        if self.orientations is not None and not np.array_equal(self.orientations, other.orientations):
            print(f"  [__eq__] orientations not equal")
            return False
        return True

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    @typechecked
    def from_ros2_bag(cls, bag_path: Union[Path, str], imu_topic: str, frame_id: str):
        """
        Creates a class structure from a ROS2 bag file with an Imu topic.

        Args:
            bag_path (Path | str): Path to the ROS2 bag file.
            imu_topic (str): Topic of the Imu messages.
            frame_id (str): The frame where this IMU data was collected.
        Returns:
            ImuData: Instance of this class.
        """

        print("WARNING: This code does not check the orientation covariance to determine if the orientation is valid; may use invalid orientations!")

        # Get topic message count and typestore
        bag_wrapper = Ros2BagWrapper(bag_path, None)
        typestore: Typestore = bag_wrapper.get_typestore()
        num_msgs: int = bag_wrapper.get_topic_count(imu_topic)

        # TODO: Load the frame id directly from the ROS2 bag.

        # Setup arrays to hold data
        timestamps = np.zeros((num_msgs), dtype=object)
        lin_acc = np.zeros((num_msgs, 3), dtype=np.double)
        ang_vel = np.zeros((num_msgs, 3), dtype=np.double)
        orientation = np.zeros((num_msgs, 4), dtype=np.double)

        # Extract the images/timestamps and save
        with Reader2(bag_path) as reader: 
            i = 0
            connections = [x for x in reader.connections if x.topic == imu_topic]
            for conn, _, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

                # Extract imu data 
                lin_acc[i] = np.array([msg.linear_acceleration.x, 
                                       msg.linear_acceleration.y, 
                                       msg.linear_acceleration.z], dtype=np.double)
                ang_vel[i] = np.array([msg.angular_velocity.x, 
                                       msg.angular_velocity.y, 
                                       msg.angular_velocity.z], dtype=np.double)
                orientation[i] = np.array([msg.orientation.x, 
                                       msg.orientation.y, 
                                       msg.orientation.z,
                                       msg.orientation.w], dtype=np.double)

                # Extract timestamps
                timestamps[i] = Ros2BagWrapper.extract_timestamp(msg)

                # Update the count
                i += 1

        # Create an ImageData class
        return cls(frame_id, CoordinateFrame.FLU, timestamps, lin_acc, ang_vel, orientation)

    @classmethod
    @typechecked
    def from_ros1_bag(cls, bag_path: Union[Path, str], imu_topic: str, frame_id: str):
        """
        Creates a class structure from a ROS1 bag file with an Imu topic.

        Args:
            bag_path (Path | str): Path to the ROS1 .bag file.
            imu_topic (str): Topic of the sensor_msgs/Imu messages.
            frame_id (str): The frame where this IMU data was collected.
        Returns:
            ImuData: Instance of this class.
        Raises:
            ValueError: If ``imu_topic`` is not present in the bag.
        """

        print("WARNING: This code does not check the orientation covariance to determine if the orientation is valid; may use invalid orientations!")

        # Get typestore and topic message count
        typestore = get_typestore(Stores.ROS1_NOETIC)

        # TODO: Load the frame id directly from the ROS1 bag.

        with Reader1(Path(bag_path)) as reader:
            conns = [c for c in reader.connections if c.topic == imu_topic]
            if not conns:
                raise ValueError(f"Topic {imu_topic!r} not found in bag {bag_path}.")
            conn = conns[0]
            num_msgs = len(reader.indexes[conn.id])

            # Setup arrays to hold data
            timestamps_np = np.zeros(num_msgs, dtype=Decimal)
            lin_acc_np = np.zeros((num_msgs, 3), dtype=np.double)
            ang_vel_np = np.zeros((num_msgs, 3), dtype=np.double)
            orientation_np = np.zeros((num_msgs, 4), dtype=np.double)

            # Extract the imu data/timestamps and save
            pbar = tqdm.tqdm(total=num_msgs, desc="Extracting IMU...", unit=" msgs")
            for i, (_, _, rawdata) in enumerate(reader.messages(connections=conns)):
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

                # Extract imu data
                lin_acc_np[i] = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                                          msg.linear_acceleration.z], dtype=np.double)
                ang_vel_np[i] = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                                          msg.angular_velocity.z], dtype=np.double)
                orientation_np[i] = np.array([msg.orientation.x, msg.orientation.y,
                                              msg.orientation.z, msg.orientation.w], dtype=np.double)

                # Extract timestamps
                timestamps_np[i] = (Decimal(msg.header.stamp.sec) +
                                    Decimal(msg.header.stamp.nanosec) * Decimal('1e-9'))

                # Update the count
                pbar.update(1)

        # Create an ImuData class
        return cls(frame_id, CoordinateFrame.FLU, timestamps_np, lin_acc_np, ang_vel_np, orientation_np)

    @classmethod
    @typechecked
    def from_txt(cls, file_path: Union[Path, str], frame_id: str, frame: CoordinateFrame, nine_axis: bool = False):
        """
        Creates a class structure from the TartanAir dataset format, which includes
        various .txt files with IMU data. It expects (in order) the timestamp, the linear
        acceleration, and the angular velocity seperated by spaces. Will also expect 
        orientation (xyzw) at the end if nine_axis is set to True.

        Args:
            file_path (Path | str): Path to the file containing the IMU data.
            frame_id (str): The frame where this IMU data was collected.
            frame (CoordinateFrame): The coordinate system convention of this data.
            nine_axis (bool): If true, will also load orientations from txt file.
        Returns:
            ImuData: Instance of this class.
        """
        
        # Count the number of lines in the file
        line_count = 0
        with open(str(file_path), 'r') as file:
            for _ in file: 
                line_count += 1

        # Setup arrays to hold data
        timestamps = np.zeros((line_count), dtype=object)
        lin_acc = np.zeros((line_count, 3), dtype=object)
        ang_vel = np.zeros((line_count, 3), dtype=object)
        if nine_axis:
            orientations = np.zeros((line_count, 4), dtype=object)
        else:
            orientations = None

        # Open the txt file and read in the data
        with open(str(file_path), 'r') as file:
            for i, line in enumerate(file):
                line_split = line.split(' ')
                timestamps[i] = line_split[0]
                lin_acc[i] = line_split[1:4]
                ang_vel[i] = line_split[4:7]
                if nine_axis:
                    orientations[i] = line_split[7:11]
        
        # Create the ImuData class
        return cls(frame_id, frame, timestamps, lin_acc, ang_vel, orientations)
    
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
        self.lin_acc = self.lin_acc[mask]
        self.ang_vel = self.ang_vel[mask]
        if self.orientations is not None:
            self.orientations = self.orientations[mask]

    # =========================================================================
    # ============================ Visualization ============================== 
    # =========================================================================  

    def visualize(self, ts_start: float, ts_end: float):
        """
        Plot the linear acceleration, angular velocity, and orientation data.

        Args:
            ts_start: Start timestamp of the window to plot.
            ts_end: End timestamp of the window to plot.
        """

        def multi_list_plotter(data: np.ndarray, timestamps: np.ndarray, title: str, keys: List[str], ylabel: str):
            """
            Helper function that plots each list in a dictionary in its own subplot,
            all in a single matplotlib figure.
            """

            num_plots = data.shape[1]
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)

            # Make it a list always
            if num_plots == 1:
                axes = [axes]
            timestamps_shifted = timestamps - timestamps[0]

            for i in range(0, num_plots):
                axes[i].plot(timestamps_shifted, data[:,i], label=keys[i])
                axes[i].set_ylabel(f"{keys[i]} - ({ylabel})")
                axes[i].grid(True)
                axes[i].legend(loc='upper right')

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{title} vs Time", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            plt.close(fig)

        # Create all plots
        mask = (self.timestamps >= ts_start) & (self.timestamps <= ts_end)
        timestamps_float = dec_arr_to_float_arr(self.timestamps[mask])
        multi_list_plotter(dec_arr_to_float_arr(self.lin_acc[mask]), timestamps_float, "Linear Acceleration", ['x', 'y', 'z'], "m/s^2")
        multi_list_plotter(dec_arr_to_float_arr(self.ang_vel[mask]), timestamps_float, "Angular Velocity", ['roll', 'pitch', 'yaw'], "rad/s")
        multi_list_plotter(dec_arr_to_float_arr(self.orientations[mask]), timestamps_float, "Orientation", ['x', 'y', 'z', 'w'], "units")
        
    # =========================================================================
    # ============================ Export Methods ============================= 
    # =========================================================================  

    def to_PathData(self, initial_pos: NDArray, initial_vel: NDArray, 
                    initial_ori: NDArray, use_ang_vel: bool) -> PathData:
        """
        Converts this IMUData class into OdometeryData by integrating the IMU data using
        Euler's method.

        Parameters:
            initial_pos: The initial position as a numpy array.
            initial_vel: The initial velocity as a numpy array.
            initial_ori: The initial orientation as a numpy array (quaternion x, y, z, w),
                only used if use_ang_vel is True.
            use_ang_vel: If True, will use angular velocity data to calculate orientation.
                If False, will use orientation data directly from the IMUData class.

        Returns:
            PathData: The resulting PathData class.
        """

        print("WARNING: This code has not been extensively tested yet!")

        # Setup arrays to hold integrated data 
        pos = np.zeros((self.len(), 3), dtype=float)
        pos[0] = initial_pos
        vel = np.zeros((self.len(), 3), dtype=float)
        vel[0] = initial_vel

        # Setup array to hold orientation data
        if use_ang_vel:
            ori = np.zeros((self.len(), 4), dtype=float)  
            ori[:, 3] = np.ones((self.len()), dtype=float)
            ori[0] = initial_ori
        else:
            if self.orientations is not None:
                ori = dec_arr_to_float_arr(self.orientations)
            else:
                raise ValueError("use_ang_vel is False, but this ImuData hs no orientation data to use!")

        # Setup a tqdm progress bar
        pbar = tqdm.tqdm(total=self.len()-1, desc="Integrating IMU Data", unit="steps")

        # Integrate the IMU data
        for i in range(1, self.len()):
            # Get time difference
            dt: float = dec_arr_to_float_arr(np.array([self.timestamps[i] - self.timestamps[i-1]])).item()

            # Calculate orientation
            if use_ang_vel:
                cur_ori = R.from_quat(ori[i-1])
                delta_q = R.from_rotvec(dec_arr_to_float_arr(self.ang_vel[i-1]) * dt)
                new_ori = (cur_ori * delta_q).as_quat()
                ori[i] = new_ori / np.linalg.norm(new_ori)

            # Rotate linear acceleration into world frame
            r = R.from_quat(ori[i-1])
            lin_acc_world = r.apply(dec_arr_to_float_arr(self.lin_acc[i-1]))
            lin_acc_world = lin_acc_world

            # Subtract gravity
            GRAVITY_CONST = 9.80665
            if self.frame == CoordinateFrame.FLU:
                lin_acc_world[2] -= GRAVITY_CONST
            elif self.frame == CoordinateFrame.NED:
                lin_acc_world[2] += GRAVITY_CONST
            else:
                raise RuntimeError(f"to_PathData() doesn't currently support this frame: {self.frame}!")

            # Calculate velocity
            vel[i] = vel[i-1] + lin_acc_world * dt

            # Calculate position
            pos[i] = pos[i-1] + vel[i-1] * dt + 0.5 * lin_acc_world * dt * dt
            pbar.update(1)

        # Return the resulting PathData class
        return PathData(self.frame_id, self.timestamps, pos, ori, self.frame)

    # =========================================================================
    # =========================== Conversion to ROS =========================== 
    # ========================================================================= 

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """
        Return the ROS message type class for an Imu message.

        Args:
            lib_type: Which ROS message library to use.

        Returns:
            The ROS message type class for ``sensor_msgs/Imu``.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            return typestore.types['sensor_msgs/msg/Imu'].__msgtype__
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            return ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Imu')
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for ImuData.get_ros_msg_type()!")
            
    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int):
        """
        Gets an Imu ROS message corresponding to the data at index i.

        Args:
            lib_type: Which ROS message library to use.
            i: The index of the IMU data to convert.

        Raises:
            ValueError: If ``i`` is outside the data bounds.
            NotImplementedError: If ``lib_type`` is not supported.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = (self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN)

        # Write the data into the new msg
        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            Imu = typestore.types['sensor_msgs/msg/Imu']
            Header = typestore.types['std_msgs/msg/Header']
            Time = typestore.types['builtin_interfaces/msg/Time']
            Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
            Vector3 = typestore.types['geometry_msgs/msg/Vector3']

            if self.orientations is not None:
                ori = Quaternion(x=self.orientations[i][0], y=self.orientations[i][1],
                           z=self.orientations[i][2], w=self.orientations[i][3])
                ori_cov = np.zeros(9, dtype=np.float64)
            else:
                ori = Quaternion(x=0, y=0, z=0, w=1)
                ori_cov = np.zeros(9, dtype=np.float64)
                ori_cov[0] = -1

            return Imu(Header(stamp=Time(sec=int(seconds), 
                                        nanosec=int(nanoseconds)), 
                            frame_id=self.frame_id),
                        orientation=ori,
                        orientation_covariance=ori_cov,
                        angular_velocity=Vector3(x=self.ang_vel[i][0],
                                                y=self.ang_vel[i][1],
                                                z=self.ang_vel[i][2]),
                        angular_velocity_covariance=np.zeros(9),
                        linear_acceleration=Vector3(x=self.lin_acc[i][0],
                                                    y=self.lin_acc[i][1], 
                                                    z=self.lin_acc[i][2]),
                        linear_acceleration_covariance=np.zeros(9))
        
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')
            Imu = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Imu')
            Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')
            Vector3 = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Vector3')

            # Create the message objects
            imu_msg = Imu()
            imu_msg.header = Header()
            if lib_type == ROSMsgLibType.RCLPY: 
                Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                imu_msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
            else:
                rospy = ModuleImporter.get_module('rospy')
                imu_msg.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))

            # Populate the rest of the data
            imu_msg.header.frame_id = self.frame_id
            imu_msg.orientation = Quaternion()
            if self.orientations is not None:
                imu_msg.orientation.x = float(self.orientations[i][0])  
                imu_msg.orientation.y = float(self.orientations[i][1])
                imu_msg.orientation.z = float(self.orientations[i][2])
                imu_msg.orientation.w = float(self.orientations[i][3])
                imu_msg.orientation_covariance = np.zeros(9, dtype=np.float64)
            else:
                imu_msg.orientation.x = 0.0
                imu_msg.orientation.y = 0.0
                imu_msg.orientation.z = 0.0
                imu_msg.orientation.w = 1.0
                covariance = np.zeros(9, dtype=np.float64)
                covariance[0] = -1
                imu_msg.orientation_covariance = covariance

            imu_msg.angular_velocity = Vector3()
            imu_msg.angular_velocity.x = float(self.ang_vel[i][0])
            imu_msg.angular_velocity.y = float(self.ang_vel[i][1])
            imu_msg.angular_velocity.z = float(self.ang_vel[i][2])
            imu_msg.angular_velocity_covariance = np.zeros(9)
            imu_msg.linear_acceleration = Vector3()
            imu_msg.linear_acceleration.x = float(self.lin_acc[i][0])
            imu_msg.linear_acceleration.y = float(self.lin_acc[i][1])
            imu_msg.linear_acceleration.z = float(self.lin_acc[i][2])
            imu_msg.linear_acceleration_covariance = np.zeros(9)
            return imu_msg

        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for ImuData.get_ros_msg()!")

    # =========================================================================
    # ========================= Multi ImuData Methods ==========================
    # =========================================================================

    @staticmethod
    def crop_to_matched(data1: 'ImuData', data2: 'ImuData', tolerance: Decimal) -> None:
        """
        Crop two ImuData objects in place so only mutually-matched entries
        remain.

        Args:
            data1: The first ImuData object, cropped in place.
            data2: The second ImuData object, cropped in place.
            tolerance: Maximum allowed absolute time difference between
                matched timestamps.

        Raises:
            NotImplementedError: Always; must be overridden with real logic.
        """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
    
from __future__ import annotations

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from .Data import CoordinateFrame, ROSMsgLibType
import decimal
from decimal import Decimal
from evo.core import geometry
import matplotlib.pyplot as plt
from ..ModuleImporter import ModuleImporter
import numpy as np
from numpy.typing import NDArray
from .PathData import PathData
from pathlib import Path
from ..ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
from typing import Union, List, Tuple, Any
import tqdm

PATH_SLICE_STEP = 40

@typechecked
class OdometryData(PathData):
    """
    Odometry data extending PathData with a child frame ID and ROS message caching.

    Supports loading from ROS2 bags, CSV files, and TXT files, and exporting
    to CSV or ROS messages (Odometry, Path, and maplab OdometryWithImuBiases).

    Attributes:
        child_frame_id: The frame whose pose is described by this odometry (e.g. ``"base_link"``).
        poses: Cached rosbags PoseStamped messages (rebuilt after any mutation).
        poses_rclpy: Cached rclpy/rospy PoseStamped messages (rebuilt after any mutation).
    """

    # Define odometry-specific data attributes
    child_frame_id: str
    poses: list # Saved nav_msgs/msg/Pose for rosbags
    poses_rclpy: list # Saved nav_msgs/msg/Pose for rclpy

    def __init__(self, frame_id: str, child_frame_id: str, timestamps: Union[np.ndarray, list], 
                 positions: Union[np.ndarray, list], orientations: Union[np.ndarray, list], frame: CoordinateFrame):
        
        # Copy initial values into attributes
        super().__init__(frame_id, timestamps, positions, orientations, frame)
        self.child_frame_id: str = child_frame_id
        self.poses: list = []
        self.poses_rclpy: list = []

        # Check to ensure that all arrays have same length
        if len(self.timestamps) != len(self.positions) or len(self.positions) != len(self.orientations):
            raise ValueError("Lengths of timestamp, position, and orientation arrays are not equal!")

    def _invalidate_cache(self):
        """ Clears cached ROS message data after mutations. """
        self.poses = []
        self.poses_rclpy = []

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_ros2_bag(cls, bag_path: Union[Path, str], odom_topic: str, frame: CoordinateFrame):
        """
        Creates a class structure from a ROS2 bag file with an Odometry topic.

        Args:
            bag_path (Path | str): Path to the ROS2 bag file.
            odom_topic (str): Topic of the Odometry messages.
        Returns:
            OdometryData: Instance of this class.
        """

        # Get topic message count and typestore
        bag_wrapper = Ros2BagWrapper(bag_path, None)
        typestore: Typestore = bag_wrapper.get_typestore()
        num_msgs: int = bag_wrapper.get_topic_count(odom_topic)
        
        # Pre-allocate arrays (memory-mapped or otherwise)
        timestamps_np = np.zeros(num_msgs, dtype=Decimal)
        positions_np = np.zeros((num_msgs, 3), dtype=Decimal)
        orientations_np = np.zeros((num_msgs, 4), dtype=Decimal)

        # Setup tqdm bar & counter
        pbar = tqdm.tqdm(total=num_msgs, desc="Extracting Odometry...", unit=" msgs")
        i = 0

        # Extract the odometry information
        frame_id, child_frame_id = None, None
        with Reader2(str(bag_path)) as reader:

            # Extract frames from first message
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):  
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                frame_id = msg.header.frame_id
                child_frame_id = msg.child_frame_id
                break

            # Extract individual message data
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                
                timestamps_np[i] = bag_wrapper.extract_timestamp(msg)
                pos = msg.pose.pose.position
                positions_np[i] = np.array([Decimal(pos.x), Decimal(pos.y), Decimal(pos.z)])
                ori = msg.pose.pose.orientation
                orientations_np[i] = np.array([Decimal(ori.x), Decimal(ori.y), Decimal(ori.z), Decimal(ori.w)])

                # NOTE: Doesn't support Twist information currently

                # Increment the count
                i += 1
                pbar.update(1)

        # Create an OdometryData class
        return cls(frame_id, child_frame_id, timestamps_np, positions_np, orientations_np, frame)
    
    @classmethod
    def from_csv(cls, csv_path: Union[Path, str], frame_id: str, child_frame_id: str, frame: CoordinateFrame,
                 header_included: bool, column_to_data: Union[List[int], None] = None,
                 separator: Union[str, None] = None, filter: Union[Tuple[str, str], None] = None,
                 ts_in_ns: bool = False, reorder_data: bool = False):
        """
        Creates a class structure from a csv file.

        Args:
            csv_path (Path | str): Path to the CSV file.
            frame_id (str): The frame that this odometry is relative to.
            child_frame_id (str): The frame whose pose is represented by this odometry.
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
            OdometryData: Instance of this class.
        """

        path_data = super().from_csv(csv_path, frame_id, frame, header_included, column_to_data,
                                     separator, filter, ts_in_ns, reorder_data)
        return cls(frame_id, child_frame_id, path_data.timestamps, path_data.positions, path_data.orientations, frame)
    
    @classmethod
    def from_txt(cls, file_path: Union[Path, str], frame_id: str, child_frame_id: str, frame: CoordinateFrame,
                 header_included: bool, column_to_data: Union[List[int], None] = None):
        """
        Creates an OdometryData class from a text file.

        Args:
            file_path (Path | str): Path to the file containing the odometry data.
            frame_id (str): The frame where this odometry is relative to.
            child_frame_id (str): The frame whose pose is represented by this odometry.
            frame (CoordinateFrame): The coordinate system convention of this data.
            header_included (bool): If this text file has a header, so we can remove it.
            column_to_data (list[int]): Tells the algorithms which columns in the text file contain which
                of the following data: ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']. Thus,
                index 0 of column_to_data should be the column that timestamp data is found in the
                text file. Set to None to use [0,1,2,3,4,5,6,7].
        Returns:
            OdometryData: Instance of this class.
        """

        path_data = super().from_txt(file_path, frame_id, frame, header_included, column_to_data)
        return cls(frame_id, child_frame_id, path_data.timestamps, path_data.positions, path_data.orientations, frame)

    @classmethod
    def from_tum(cls, file_path: Union[Path, str], frame_id: str, child_frame_id: str, frame: CoordinateFrame):
        """
        Creates an OdometryData class from a TUM RGB-D dataset trajectory format text file.

        Each row must contain 8 space-separated values::

            timestamp x y z q_x q_y q_z q_w

        where ``timestamp`` is in seconds and the orientation quaternion is in
        ``(x, y, z, w)`` order.

        Args:
            file_path (Path | str): Path to the TUM trajectory file.
            frame_id (str): The frame where this odometry is relative to.
            child_frame_id (str): The frame whose pose is represented by this odometry.
            frame (CoordinateFrame): The coordinate system convention of this data.

        Returns:
            OdometryData: Instance of this class.
        """
        # TUM order: ts x y z qx qy qz qw
        # column_to_data: ts=0, x=1, y=2, z=3, qw=7, qx=4, qy=5, qz=6
        return cls.from_txt(file_path, frame_id, child_frame_id, frame,
                            header_included=False,
                            column_to_data=[0, 1, 2, 3, 7, 4, 5, 6])
    
    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType, msg_type: str = "Odometry") -> Any:
        """
        Return the ROS message type class for odometry-related messages.

        Args:
            lib_type: Which ROS message library to use.
            msg_type: The message type name. Supported values are ``"Odometry"``,
                ``"Path"``, ``"TFMessage"``, and ``"maplab_msg/OdometryWithImuBiases"``
                (ROSPY only).

        Returns:
            The ROS message type class.

        Raises:
            ValueError: If ``msg_type`` is not supported for the given ``lib_type``.
            NotImplementedError: If ``lib_type`` is not supported.
        """
        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            if msg_type == "Odometry":
                return typestore.types['nav_msgs/msg/Odometry'].__msgtype__
            elif msg_type == "Path":
                return typestore.types['nav_msgs/msg/Path'].__msgtype__
            elif msg_type == "TFMessage":
                return typestore.types['tf2_msgs/msg/TFMessage'].__msgtype__
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type}")
        elif lib_type == ROSMsgLibType.RCLPY:
            if msg_type == "Path":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
            elif msg_type == "TFMessage":
                return ModuleImporter.get_module_attribute('tf2_msgs.msg', 'TFMessage')
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type}")
        elif lib_type == ROSMsgLibType.ROSPY:
            if msg_type == "Odometry":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Odometry')
            if msg_type == "maplab_msg/OdometryWithImuBiases":
                return ModuleImporter.get_module_attribute('maplab_msgs.msg', 'OdometryWithImuBiases')
            elif msg_type == "Path":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
            elif msg_type == "TFMessage":
                return ModuleImporter.get_module_attribute('tf2_msgs.msg', 'TFMessage')
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type}")
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for OdometryData.get_ros_msg_type()!")
    
    def _extract_seconds_and_nanoseconds(self, i: int):
        seconds = int(self.timestamps[i])
        nanoseconds = (self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) \
                        * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN)
        return seconds, nanoseconds
    
    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int, msg_type: str = "Odometry"):
        """
        Gets an Odometry ROS message corresponding to the data at index i.

        Args:
            lib_type: Which ROS message library to use.
            i: The index of the odometry data to convert.
            msg_type: The message type name (``"Odometry"``, ``"Path"``,
                ``"TFMessage"``, or ``"maplab_msg/OdometryWithImuBiases"``).

        Raises:
            IndexError: If ``i`` is outside the data bounds.
            ValueError: If ``msg_type`` is not supported for the given ``lib_type``.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise IndexError(f"Index {i} is out of bounds!")
        
        # Extract seconds and nanoseconds
        seconds, nanoseconds = self._extract_seconds_and_nanoseconds(i)

        # Write the data into the new msg
        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)

            Odometry = typestore.types['nav_msgs/msg/Odometry']
            Header = typestore.types['std_msgs/msg/Header']
            Time = typestore.types['builtin_interfaces/msg/Time']
            PoseWithCovariance = typestore.types['geometry_msgs/msg/PoseWithCovariance']
            TwistWithCovariance = typestore.types['geometry_msgs/msg/TwistWithCovariance']
            Twist = typestore.types['geometry_msgs/msg/Twist']
            Vector3 = typestore.types['geometry_msgs/msg/Vector3']
            Path = typestore.types['nav_msgs/msg/Path']
            Pose = typestore.types['geometry_msgs/msg/Pose']
            Point = typestore.types['geometry_msgs/msg/Point']
            Quaternion = typestore.types['geometry_msgs/msg/Quaternion']

            if msg_type == "Odometry":
            
                return Odometry(Header(stamp=Time(sec=int(seconds), 
                                                nanosec=int(nanoseconds)), 
                                frame_id=self.frame_id),
                                child_frame_id=self.child_frame_id,
                                pose=PoseWithCovariance(pose=Pose(position=Point(x=self.positions[i][0],
                                                                    y=self.positions[i][1],
                                                                    z=self.positions[i][2]),
                                                        orientation=Quaternion(x=self.orientations[i][0],
                                                                                y=self.orientations[i][1],
                                                                                z=self.orientations[i][2],
                                                                                w=self.orientations[i][3])),
                                                        covariance=np.zeros(36)),
                                twist=TwistWithCovariance(twist=Twist(linear=Vector3(x=0, # Currently doesn't support Twist
                                                                                    y=0,
                                                                                    z=0,),
                                                                    angular=Vector3(x=0,
                                                                                    y=0,
                                                                                    z=0,)),
                                                        covariance=np.zeros(36)))
            elif msg_type == "Path":

                # Pre-calculate all the poses
                if len(self.poses) != self.len():
                    PoseStamped = typestore.types['geometry_msgs/msg/PoseStamped']

                    for j in range(self.len()):
                        sec, nanosec = self._extract_seconds_and_nanoseconds(j)
                        self.poses.append(PoseStamped(Header(stamp=Time(sec=int(sec), 
                                                                        nanosec=int(nanosec)),
                                                            frame_id=self.frame_id),
                                                    pose=Pose(position=Point(x=self.positions[j][0],
                                                                            y=self.positions[j][1],
                                                                            z=self.positions[j][2]),
                                    orientation=Quaternion(x=self.orientations[j][0],
                                                            y=self.orientations[j][1],
                                                            z=self.orientations[j][2],
                                                            w=self.orientations[j][3]))))

                return Path(Header(stamp=Time(sec=int(seconds),
                                            nanosec=int(nanoseconds)),
                                frame_id=self.frame_id),
                                poses=self.poses[0:i+1:PATH_SLICE_STEP])
            elif msg_type == "TFMessage":

                TFMessage = typestore.types['tf2_msgs/msg/TFMessage']
                TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
                Transform = typestore.types['geometry_msgs/msg/Transform']
                Vector3 = typestore.types['geometry_msgs/msg/Vector3']

                return TFMessage(transforms=[
                    TransformStamped(
                        header=Header(stamp=Time(sec=int(seconds), nanosec=int(nanoseconds)),
                                      frame_id=self.frame_id),
                        child_frame_id=self.child_frame_id,
                        transform=Transform(
                            translation=Vector3(x=self.positions[i][0],
                                                y=self.positions[i][1],
                                                z=self.positions[i][2]),
                            rotation=Quaternion(x=self.orientations[i][0],
                                                y=self.orientations[i][1],
                                                z=self.orientations[i][2],
                                                w=self.orientations[i][3])))])
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type} with ROSMsgLibType.ROSBAGS")
            
        elif lib_type == ROSMsgLibType.ROSPY or lib_type == ROSMsgLibType.RCLPY:
            if msg_type == "Odometry":

                Odometry = ModuleImporter.get_module_attribute('nav_msgs.msg', 'Odometry')
                Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')
                PoseWithCovariance = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'PoseWithCovariance')
                TwistWithCovariance = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'TwistWithCovariance')
                Pose = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Pose')
                Point = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Point')
                Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')
                Vector3 = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Vector3')

                msg = Odometry()
                msg.header = Header()
                msg.header.frame_id = self.frame_id
                if lib_type == ROSMsgLibType.RCLPY: 
                    Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                    msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
                else:
                    rospy = ModuleImporter.get_module('rospy')
                    msg.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))
                msg.child_frame_id = self.child_frame_id

                msg.pose = PoseWithCovariance()
                msg.pose.pose = Pose()
                msg.pose.pose.position = Point()
                msg.pose.pose.position.x = float(self.positions[i][0])
                msg.pose.pose.position.y = float(self.positions[i][1])
                msg.pose.pose.position.z = float(self.positions[i][2])
                msg.pose.pose.orientation = Quaternion()
                msg.pose.pose.orientation.x = float(self.orientations[i][0])
                msg.pose.pose.orientation.y = float(self.orientations[i][1])
                msg.pose.pose.orientation.z = float(self.orientations[i][2])
                msg.pose.pose.orientation.w = float(self.orientations[i][3])
                msg.pose.covariance = np.zeros(36) # NOTE: Assumes covariance of zero.
                msg.twist = TwistWithCovariance()
                msg.twist.twist.linear = Vector3()
                msg.twist.twist.linear.x = 0.0  # NOTE: Currently doesn't support Twist
                msg.twist.twist.linear.y = 0.0
                msg.twist.twist.linear.z = 0.0
                msg.twist.twist.angular = Vector3()
                msg.twist.twist.angular.x = 0.0
                msg.twist.twist.angular.y = 0.0
                msg.twist.twist.angular.z = 0.0
                msg.twist.covariance = np.zeros(36)
                return msg
            
            elif msg_type == "maplab_msg/OdometryWithImuBiases":
                
                if lib_type == ROSMsgLibType.RCLPY:
                    raise ValueError("maplab_msg/OdometryWithImuBiases is not supported for RCLPY!")

                rospy = ModuleImporter.get_module('rospy')
                OdometryWithImuBiases = ModuleImporter.get_module_attribute('maplab_msgs.msg', 'OdometryWithImuBiases')
                PoseWithCovariance = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'PoseWithCovariance')
                TwistWithCovariance = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'TwistWithCovariance')
                Point = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Point')
                Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')
                Vector3 = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Vector3')
                Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')

                msg = OdometryWithImuBiases()
                msg.header = Header()
                msg.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))
                msg.header.frame_id = self.frame_id
                msg.child_frame_id = self.child_frame_id
                msg.pose = PoseWithCovariance()
                msg.pose.pose.position = Point()
                msg.pose.pose.position.x = float(self.positions[i][0])
                msg.pose.pose.position.y = float(self.positions[i][1])
                msg.pose.pose.position.z = float(self.positions[i][2])
                msg.pose.pose.orientation = Quaternion()
                msg.pose.pose.orientation.x = float(self.orientations[i][0])
                msg.pose.pose.orientation.y = float(self.orientations[i][1])
                msg.pose.pose.orientation.z = float(self.orientations[i][2])
                msg.pose.pose.orientation.w = float(self.orientations[i][3])
                msg.pose.covariance = np.zeros(36) # NOTE: Assumes covariance of zero.
                msg.twist = TwistWithCovariance()
                msg.twist.twist.linear = Vector3()
                msg.twist.twist.linear.x = 0.0  # NOTE: Currently doesn't support Twist
                msg.twist.twist.linear.y = 0.0
                msg.twist.twist.linear.z = 0.0
                msg.twist.twist.angular = Vector3()
                msg.twist.twist.angular.x = 0.0
                msg.twist.twist.angular.y = 0.0
                msg.twist.twist.angular.z = 0.0
                msg.twist.covariance = np.zeros(36)
                msg.accel_bias = Vector3() # NOTE: Assumes IMU biases are zero
                msg.accel_bias.x = 0.0
                msg.accel_bias.y = 0.0
                msg.accel_bias.z = 0.0
                msg.gyro_bias = Vector3()
                msg.gyro_bias.x = 0.0
                msg.gyro_bias.y = 0.0
                msg.gyro_bias.z = 0.0
                msg.odometry_state = 0 # NOTE: Assumes default state
                return msg

            elif msg_type == "Path":
                
                Path = ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
                Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')

                # Pre-calculate all the poses
                if len(self.poses_rclpy) != self.len():

                    PoseStamped = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'PoseStamped')
                    Pose = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Pose')
                    Point = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Point')
                    Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')

                    for j in range(self.len()):
                        sec, nanosec = self._extract_seconds_and_nanoseconds(j)
                        pose_msg = PoseStamped()
                        pose_msg.header = Header()
                        if lib_type == ROSMsgLibType.RCLPY:
                            Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                            pose_msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
                        else:
                            rospy = ModuleImporter.get_module('rospy')
                            pose_msg.header.stamp = rospy.Time(secs=int(seconds), nsecs=int(nanoseconds))
                        pose_msg.header.frame_id = self.frame_id
                        pose_msg.pose = Pose()
                        pose_msg.pose.position = Point()
                        pose_msg.pose.position.x = float(self.positions[j][0])
                        pose_msg.pose.position.y = float(self.positions[j][1])
                        pose_msg.pose.position.z = float(self.positions[j][2])
                        pose_msg.pose.orientation = Quaternion()
                        pose_msg.pose.orientation.x = float(self.orientations[j][0])
                        pose_msg.pose.orientation.y = float(self.orientations[j][1])
                        pose_msg.pose.orientation.z = float(self.orientations[j][2])
                        pose_msg.pose.orientation.w = float(self.orientations[j][3])
                        self.poses_rclpy.append(pose_msg)

                msg = Path()
                msg.header = Header()
                if lib_type == ROSMsgLibType.RCLPY:
                    Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                    msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
                else:
                    rospy = ModuleImporter.get_module('rospy')
                    msg.header.stamp = rospy.Time(secs=int(seconds), nsecs=int(nanoseconds))
                msg.header.frame_id = self.frame_id
                msg.poses = self.poses_rclpy[0:i+1:PATH_SLICE_STEP]
                return msg

            elif msg_type == "TFMessage":

                TFMessage = ModuleImporter.get_module_attribute('tf2_msgs.msg', 'TFMessage')
                TransformStamped = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'TransformStamped')
                Transform = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Transform')
                Vector3 = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Vector3')
                Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')
                Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')

                ts = TransformStamped()
                ts.header = Header()
                if lib_type == ROSMsgLibType.RCLPY:
                    Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                    ts.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
                else:
                    rospy = ModuleImporter.get_module('rospy')
                    ts.header.stamp = rospy.Time(secs=seconds, nsecs=int(nanoseconds))
                ts.header.frame_id = self.frame_id
                ts.child_frame_id = self.child_frame_id
                ts.transform = Transform()
                ts.transform.translation = Vector3()
                ts.transform.translation.x = float(self.positions[i][0])
                ts.transform.translation.y = float(self.positions[i][1])
                ts.transform.translation.z = float(self.positions[i][2])
                ts.transform.rotation = Quaternion()
                ts.transform.rotation.x = float(self.orientations[i][0])
                ts.transform.rotation.y = float(self.orientations[i][1])
                ts.transform.rotation.z = float(self.orientations[i][2])
                ts.transform.rotation.w = float(self.orientations[i][3])

                msg = TFMessage()
                msg.transforms = [ts]
                return msg

            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type} with ROSMsgLibType.ROSPY or ROSMsgLibType.RCLPY.")
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for OdometryData.get_ros_msg()!")
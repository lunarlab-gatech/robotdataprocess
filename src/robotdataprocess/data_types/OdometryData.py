from __future__ import annotations

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
import csv
from .Data import CoordinateFrame, ROSMsgLibType
import decimal
from decimal import Decimal
from evo.core import geometry
import matplotlib.pyplot as plt
from ..ModuleImporter import ModuleImporter
import numpy as np
from numpy.typing import NDArray
import os
import pandas as pd
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

        # Create an OdometryData class
        return cls(frame_id, child_frame_id, timestamps_np, positions_np, orientations_np, frame)
    
    @classmethod
    def from_txt_file(cls, file_path: Union[Path, str], frame_id: str, child_frame_id: str, frame: CoordinateFrame,
                      header_included: bool, column_to_data: Union[List[int], None] = None):
        """
        Creates and OdometryData class from a text file.

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
                orientations_np[i] =  np.array([line_split[column_to_data[5]], line_split[column_to_data[6]], line_split[column_to_data[7]], line_split[column_to_data[4]]])

        # Remove the header
        if header_included:
            timestamps_np = timestamps_np[1:]
            positions_np = positions_np[1:]
            orientations_np = orientations_np[1:]
        
        # Create an OdometryData class
        return cls(frame_id, child_frame_id, timestamps_np, positions_np, orientations_np, frame)
    
    # =========================================================================
    # ============================ Export Methods =============================
    # =========================================================================

    def to_csv(self, csv_path: Union[Path, str], write_header: bool = True):
        """
        Writes the odometry data to a .csv file. Note that data will be
        saved in the following order: timestamp, pos.x, pos.y, pos.z,
        ori.w, ori.x, ori.y, ori.z. Timestamp is in seconds.

        Args:
            csv_path (Path | str): Path to the output csv file.
            odom_topic (str): Topic of the Odometry messages.
            write_header (bool): If false, skip the header row.
        Returns:
            OdometryData: Instance of this class.
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

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType, msg_type: str = "Odometry") -> Any:
        """ Return the __msgtype__ for an Odometry msg. """
        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            if msg_type == "Odometry":
                return typestore.types['nav_msgs/msg/Odometry'].__msgtype__
            elif msg_type == "Path":
                return typestore.types['nav_msgs/msg/Path'].__msgtype__
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type}")
        elif lib_type == ROSMsgLibType.RCLPY:
            if msg_type == "Path":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type}")
        elif lib_type == ROSMsgLibType.ROSPY:
            if msg_type == "Odometry":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Odometry')
            if msg_type == "maplab_msg/OdometryWithImuBiases":
                return ModuleImporter.get_module_attribute('maplab_msgs.msg', 'OdometryWithImuBiases')
            elif msg_type == "Path":
                return ModuleImporter.get_module_attribute('nav_msgs.msg', 'Path')
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
        Gets an Image ROS2 Humble message corresponding to the odometry in index i.
        
        Args:
            i (int): The index of the odometry data to convert.
        Raises:
            ValueError: If i is outside the data bounds.

        NOTE: Currently doesn't support Twist information.
        NOTE: Assumes zero covariances.
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
            
            else:
                raise ValueError(f"Unsupported msg_type for OdometryData: {msg_type} with ROSMsgLibType.ROSPY or ROSMsgLibType.RCLPY.")
        else:
            raise NotImplementedError(f"Unsupported ROSMsgLibType {lib_type} for OdometryData.get_ros_msg()!")
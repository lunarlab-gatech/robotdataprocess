from ..conversion_utils import convert_collection_into_decimal_array
from .Data import Data, CoordinateFrame
import decimal
from decimal import Decimal
import numpy as np
from pathlib import Path
from ..rosbag.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked

class ImuData(Data):

    # Define IMU-specific data attributes
    lin_acc: np.ndarray[Decimal]
    ang_vel: np.ndarray[Decimal]
    orientation: np.ndarray[Decimal] # quaternions (x, y, z, w)
    frame: CoordinateFrame

    @typechecked
    def __init__(self, frame_id: str, frame: CoordinateFrame, timestamps: np.ndarray | list, 
                 lin_acc: np.ndarray | list, ang_vel: np.ndarray | list,
                 orientation: np.ndarray | list):
        
        # Copy initial values into attributes
        super().__init__(frame_id, timestamps)
        self.frame = frame
        self.lin_acc = convert_collection_into_decimal_array(lin_acc)
        self.ang_vel = convert_collection_into_decimal_array(ang_vel)
        self.orientation = convert_collection_into_decimal_array(orientation)

        # Check to ensure that all arrays have same length
        if len(self.timestamps) != len(self.lin_acc) or len(self.lin_acc) != len(self.ang_vel) \
            or len(self.ang_vel) != len(self.orientation):
            raise ValueError("Lengths of timestamp, lin_acc, ang_vel, and orientation arrays are not equal!")

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    @typechecked
    def from_ros2_bag(cls, bag_path: Path | str, imu_topic: str, frame_id: str):
        """
        Creates a class structure from a ROS2 bag file with an Imu topic.

        Args:
            bag_path (Path | str): Path to the ROS2 bag file.
            img_topic (str): Topic of the Imu messages.
            frame_id (str): The frame where this IMU data was collected.
        Returns:
            ImageData: Instance of this class.
        """

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
    def from_TartanAir(cls, folder_path: Path | str, frame_id: str):
        """
        Creates a class structure from the TartanAir dataset format, which includes
        various .txt files with IMU data.

        Args:
            folder_path (Path | str): Path to the folder containing the IMU data.
            frame_id (str): The frame where this IMU data was collected.
        Returns:
            ImuData: Instance of this class.
        """

        # Get paths to all necessary files
        ts_folder_path = Path(folder_path) / 'imu_time.npy'
        lin_acc_folder_path = Path(folder_path) / 'acc_nograv_body.npy'
        ang_vel_folder_path =  Path(folder_path) / 'gyro.npy'
        orientation_folder_path = Path(folder_path) / 'ori_global.npy'

        # Load the data
        timestamps = convert_collection_into_decimal_array(np.load(ts_folder_path))
        lin_acc = np.load(lin_acc_folder_path)
        ang_vel = np.load(ang_vel_folder_path)

        # Currently unsure of format of TartanAir Orientation data
        # (whether it's extrinsic or intrinsic euler rotations, etc.)
        # Thus, for now fill with zeros.
        orientation = np.zeros_like(ang_vel)

        # Create the ImuData class
        raise NotImplementedError("Need to know coordiante frame of TartanAir.")
        frame = None
        return cls(frame_id, frame, timestamps, lin_acc, ang_vel, orientation)
    
    @classmethod
    @typechecked
    def from_txt_file(cls, file_path: Path | str, frame_id: str, frame: CoordinateFrame):
        """
        Creates a class structure from the TartanAir dataset format, which includes
        various .txt files with IMU data. It expects the timestamp, the linear
        acceleration, and the angular velocity, seperated by spaced in that order.

        Args:
            file_path (Path | str): Path to the file containing the IMU data.
            frame_id (str): The frame where this IMU data was collected.
            frame (CoordinateFrame): The coordinate system convention of this data.
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

        # Open the txt file and read in the data
        with open(str(file_path), 'r') as file:
            for i, line in enumerate(file):
                line_split = line.split(' ')
                timestamps[i] = line_split[0]
                lin_acc[i] = line_split[1:4]
                ang_vel[i] = line_split[4:7]
        
        # Set orientation to identity, as we don't have orientation from HERCULES IMU
        orientation = np.zeros((lin_acc.shape[0], 4), dtype=int)
        orientation[:,3] = np.ones((lin_acc.shape[0]), dtype=int)

        # Create the ImuData class
        return cls(frame_id, frame, timestamps, lin_acc, ang_vel, orientation)
    
    # =========================================================================
    # ========================= Manipulation Methods ========================== 
    # =========================================================================  

    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """

        # Create boolean mask of data to keep
        mask = (self.timestamps >= start) & (self.timestamps <= end)

        # Apply mask
        self.timestamps = self.timestamps[mask]
        self.lin_acc = self.lin_acc[mask]
        self.ang_vel = self.ang_vel[mask]
        self.orientation = self.orientation[mask]

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
            # Define the rotation matrix
            R_NED = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]])
            R_NED_Q = R.from_matrix(R_NED)

            # Do a change of basis 
            raise NotImplementedError("Not sure if this should be a pose transformation or change of basis")
            self.lin_acc = (R_NED @ self.lin_acc.T).T
            self.ang_vel = (R_NED @ self.ang_vel.T).T
            for i in range(self.len()):
                self.orientation[i] = (R_NED_Q * R.from_quat(self.orientation[i]) * R_NED_Q.inv()).as_quat()

            # Update frame
            self.frame = CoordinateFrame.FLU

        # Otherwise, throw an error
        else:
            raise RuntimeError(f"ImuData class is in an unexpected frame: {self.frame}!")

    # =========================================================================
    # =========================== Conversion to ROS =========================== 
    # ========================================================================= 

    @typechecked
    @staticmethod
    def get_ros_msg_type() -> str:
        """ Return the __msgtype__ for an Imu msg. """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        return typestore.types['sensor_msgs/msg/Imu'].__msgtype__

    @typechecked
    def get_ros_msg(self, i: int):
        """
        Gets an Image ROS2 Humble message corresponding to the image represented by index i.
        
        Args:
            i (int): The index of the image message to convert.
        Raises:
            ValueError: If i is outside the data bounds.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Make sure our data is in the FLU frame, otherwise throw an error
        if self.frame != CoordinateFrame.FLU:
            raise RuntimeError("Convert this IMU Data to a FLU coordinate frame before writing to a ROS2 bag!")

        # Get ROS2 message classes
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        Imu = typestore.types['sensor_msgs/msg/Imu']
        Header = typestore.types['std_msgs/msg/Header']
        Time = typestore.types['builtin_interfaces/msg/Time']
        Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
        Vector3 = typestore.types['geometry_msgs/msg/Vector3']

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = (self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN)

        # Write the data into the new msg
        return Imu(Header(stamp=Time(sec=int(seconds), 
                                     nanosec=int(nanoseconds)), 
                          frame_id=self.frame_id),
                    orientation=Quaternion(x=0,
                                           y=0,
                                           z=0,
                                           w=1), # Currently ignores data in orientation
                    orientation_covariance=np.zeros(9),
                    angular_velocity=Vector3(x=self.ang_vel[i][0],
                                             y=self.ang_vel[i][1],
                                             z=self.ang_vel[i][2]),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=Vector3(x=self.lin_acc[i][0],
                                                y=self.lin_acc[i][1], 
                                                z=self.lin_acc[i][2]),
                    linear_acceleration_covariance=np.zeros(9))
                    
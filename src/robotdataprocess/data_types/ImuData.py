from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
from .Data import Data, CoordinateFrame
import decimal
from decimal import Decimal
import numpy as np
from pathlib import Path
from robotdataprocess.data_types.PathData import PathData
from ..rosbag.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
import tqdm

class ImuData(Data):

    # Define IMU-specific data attributes
    lin_acc: np.ndarray[Decimal]
    ang_vel: np.ndarray[Decimal]
    orientations: np.ndarray[Decimal] # quaternions (x, y, z, w)
    frame: CoordinateFrame

    @typechecked
    def __init__(self, frame_id: str, frame: CoordinateFrame, timestamps: np.ndarray | list, 
                 lin_acc: np.ndarray | list, ang_vel: np.ndarray | list,
                 orientations: np.ndarray | list):
        
        # Copy initial values into attributes
        super().__init__(frame_id, timestamps)
        self.frame = frame
        self.lin_acc = col_to_dec_arr(lin_acc)
        self.ang_vel = col_to_dec_arr(ang_vel)
        self.orientations = col_to_dec_arr(orientations)

        # Check to ensure that all arrays have same length
        if len(self.timestamps) != len(self.lin_acc) or len(self.lin_acc) != len(self.ang_vel) \
            or len(self.ang_vel) != len(self.orientations):
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
        timestamps = col_to_dec_arr(np.load(ts_folder_path))
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

        NOTE: Sets orientation to identity! 
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
        
        # Set orientation to identity, as it is assumed this text file doesn't have it
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
        self.orientations = self.orientations[mask]

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
                self.orientations[i] = (R_NED_Q * R.from_quat(self.orientations[i]) * R_NED_Q.inv()).as_quat()

            # Update frame
            self.frame = CoordinateFrame.FLU

        # Otherwise, throw an error
        else:
            raise RuntimeError(f"ImuData class is in an unexpected frame: {self.frame}!")
        
    # =========================================================================
    # ============================ Export Methods ============================= 
    # =========================================================================  

    def to_PathData(self, initial_pos: np.ndarray[float], initial_vel: np.ndarray[float], 
                    initial_ori: np.ndarray[float], use_ang_vel: bool) -> PathData:
        """
        Converts this IMUData class into OdometeryData by integrating the IMU data using
        Euler's method.

        Parameters:
            initial_pos: The initial position as a numpy array.
            initial_vel: The initial velocity as a numpy array.
            initial_ori: The initial orientation as a numpy array (quaternion x, y, z, w).
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
        else:
            ori = dec_arr_to_float_arr(self.orientations)
        ori[0] = initial_ori

        # Setup a tqdm progress bar
        pbar = tqdm.tqdm(total=self.len()-1, desc="Integrating IMU Data", unit="steps")

        # Integrate the IMU data
        for i in range(1, self.len()):
            # Get time difference
            dt: float = dec_arr_to_float_arr(self.timestamps[i] - self.timestamps[i-1])

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
                    
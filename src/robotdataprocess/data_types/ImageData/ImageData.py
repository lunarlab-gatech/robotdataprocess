from __future__ import annotations

from ..Data import Data, ROSMsgLibType
import decimal
from decimal import Decimal
from enum import Enum
from ...ModuleImporter import ModuleImporter
import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import Union, Any
from rosbags.typesys import Stores, get_typestore

@typechecked
class ImageData(Data):
    """ Generic ImageData class that should be overwritten by children """

    # Define image encodings enumeration
    class ImageEncoding(Enum):
        Mono8 = 0
        RGB8 = 1
        _32FC1 = 2

        # ================ Class Methods ================
        @classmethod
        def from_str(cls, encoding_str: str):
            if encoding_str == "ImageEncoding.Mono8":
                return cls.Mono8
            elif encoding_str == "ImageEncoding.RGB8":
                return cls.RGB8
            elif encoding_str == "ImageEncoding._32FC1":
                return cls._32FC1
            else:
                raise NotImplementedError(f"This encoding ({encoding_str}) is not yet implemented (or it doesn't exist)!")
        
        @classmethod
        def from_ros_str(cls, encoding_str: str):
            encoding_str = encoding_str.lower()
            if encoding_str == 'mono8':
                return cls.Mono8
            elif encoding_str == 'rgb8':
                return cls.RGB8
            elif encoding_str == "32fc1":
                return cls._32FC1
            else:
                raise NotImplementedError(f"This encoding ({encoding_str}) is not yet implemented (or it doesn't exist)!")
        
        @classmethod
        def from_dtype_and_channels(cls, dtype: np.dtype, channels: int):
            if dtype == np.uint8 and channels == 1:
                return cls.Mono8
            elif dtype == np.uint8 and channels == 3:
                return cls.RGB8
            elif dtype == np.float32 and channels == 1:
                return cls._32FC1
            else:
                raise NotImplementedError(f"dtype {dtype} w/ {channels} channel(s) has no corresponding encoding!")
        
        @classmethod
        def from_pillow_str(cls, encoding_str: str):
            if encoding_str == "RGB":
                return cls.RGB8
            elif encoding_str == "L":
                return cls.Mono8
            else:
                raise NotImplementedError(f"This encoding ({encoding_str}) is not yet implemented (or it doesn't exist)!")
        
        # ================ Export Methods ================
        @staticmethod
        def to_ros_str(encoding: ImageData.ImageEncoding):
            if encoding == ImageData.ImageEncoding.Mono8:
                return 'mono8'
            elif encoding == ImageData.ImageEncoding.RGB8:
                return 'rgb8'
            elif encoding == ImageData.ImageEncoding._32FC1:
                return '32FC1'
            else:
                raise NotImplementedError(f"This ImageData.ImageEncoding.{encoding} is not yet implemented (or it doesn't exist)!")
        
        @staticmethod
        def to_dtype_and_channels(encoding):
            if encoding == ImageData.ImageEncoding.Mono8:
                return (np.uint8, 1)
            elif encoding == ImageData.ImageEncoding.RGB8:
                return (np.uint8, 3)
            elif encoding == ImageData.ImageEncoding._32FC1:
                return (np.float32, 1)
            else:
                raise NotImplementedError(f"This encoding ({encoding}) is missing a mapping to dtype/channels!")
    
    height: int
    width: int
    encoding: ImageData.ImageEncoding
    images: Union[np.ndarray, Any] # With Any being a LazyImageArray

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], height: int,
                 width: int, encoding: ImageData.ImageEncoding, images: Union[np.ndarray, Any]):
        super().__init__(frame_id, timestamps)
        self.height = height
        self.width = width
        self.encoding = encoding
        self.images = images

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    def from_image_files(cls, image_folder_path: Union[Path, str], frame_id: str) -> ImageData:
        """ Creates a class structure from a folder with .png files. """
        NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    # =========================================================================
    # ========================= Manipulation Methods ========================== 
    # =========================================================================  

    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """

        # Create boolean mask of data to keep
        mask = (self.timestamps >= start) & (self.timestamps <= end)

        # Apply mask
        self.timestamps = self.timestamps[mask]
        self.images = self.images[mask]
    
    # =========================================================================
    # ======================= Multi ImageData Methods ========================= 
    # ========================================================================= 

    def compare_timestamps(self, other: ImageDataInMemory):
        """
        This method compares two ImageData objects based on the timestamps of their
        images.
        """

        # Find the locations in other where self timestamps would fit
        idxs = np.searchsorted(other.timestamps, self.timestamps, side='right')

        # Get the left indices and right indices
        idxs_right = np.clip(idxs, 0, len(other.timestamps)-1)
        idxs_left = np.clip(idxs - 1, 0, len(other.timestamps)-1)

        # Get distances to nearest on either side
        dists = np.minimum(np.abs(self.timestamps - other.timestamps[idxs_left]), 
                           np.abs(self.timestamps - other.timestamps[idxs_right]))
        
        # Print the mean and std of the distances
        print(f"Mean distance (left): {np.mean(np.abs(self.timestamps - other.timestamps[idxs_left]))}")
        print(f"Mean distance (right): {np.mean(np.abs(self.timestamps - other.timestamps[idxs_right]))}")
        print(f"Mean distance: {np.mean(dists)}")
        print(f"Std distance: {np.std(dists)}")

    # =========================================================================
    # =========================== Conversion to ROS =========================== 
    # ========================================================================= 

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """ Return the __msgtype__ for an Image msg. """

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            return typestore.types['sensor_msgs/msg/Image'].__msgtype__
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            return ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Image')
        else:
            raise NotImplementedError(f"Unsupported ROS_MSG_LIBRARY_TYPE {lib_type} for ImageData.get_ros_msg_type!")

    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int):
        """
        Gets an Image ROS2 Humble message corresponding to the image represented by index i.
        
        Args:
            lib_type (ROSMsgLibType): The type of ROS message to return (e.g., ROSBAGS, RCLPY).
            i (int): The index of the image message to convert.
        Raises:
            ValueError: If i is outside the data bounds.
        """

        # Check to make sure index is within data bounds
        if i < 0 or i >= self.len():
            raise ValueError(f"Index {i} is out of bounds!")

        # Calculate the step
        if self.encoding == ImageData.ImageEncoding.RGB8:
            step = 3 * self.width
        elif self.encoding == ImageData.ImageEncoding._32FC1:
            step = 4 * self.width
        else:
            raise NotImplementedError(f"Unsupported encoding {self.encoding} for rosbag_get_ros_msg!")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = int((self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN))

        # Calculate the ROS2 Image data
        if self.encoding == ImageData.ImageEncoding.RGB8:
            data = self.images[i].flatten()
        elif self.encoding == ImageData.ImageEncoding._32FC1:
            data = self.images[i].flatten().view(np.uint8)
            # TODO: Check endianness for _32FC1
        else:
            raise NotImplementedError(f"Unsupported encoding {self.encoding} for rosbag_get_ros_msg!")

        # Write the data into the new class
        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            Image, Header, Time = typestore.types['sensor_msgs/msg/Image'], typestore.types['std_msgs/msg/Header'], typestore.types['builtin_interfaces/msg/Time']

            return Image(Header(stamp=Time(sec=int(seconds), 
                                        nanosec=int(nanoseconds)), 
                                frame_id=self.frame_id),
                        height=self.height, 
                        width=self.width, 
                        encoding=ImageData.ImageEncoding.to_ros_str(self.encoding),
                        is_bigendian=0, 
                        step=step, 
                        data=data)
        
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')
            Image = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Image')

            # Create the messages
            img_msg = Image()
            img_msg.header = Header()
            if lib_type == ROSMsgLibType.RCLPY:
                Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                img_msg.header.stamp = Time(seconds=seconds, nanoseconds=int(nanoseconds)).to_msg()
            else:
                rospy = ModuleImporter.get_module('rospy')
                img_msg.header.stamp = rospy.Time(secs=int(seconds), nsecs=int(nanoseconds))

            # Populate the rest of the data
            img_msg.header.frame_id = self.frame_id 
            img_msg.height = self.height
            img_msg.width = self.width
            img_msg.encoding = ImageData.ImageEncoding.to_ros_str(self.encoding)
            img_msg.is_bigendian = 0
            img_msg.step = step
            img_msg.data = data.tolist()
            return img_msg

        else:
            raise NotImplementedError(f"Unsupported ROS_MSG_LIBRARY_TYPE {lib_type} for ImageData.get_ros_msg()!")
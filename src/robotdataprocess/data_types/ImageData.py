from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
import cv2
from .Data import Data
import decimal
from decimal import Decimal
from enum import Enum
import numpy as np
from numpy.lib.format import open_memmap
import os
from pathlib import Path
from PIL import Image
from ..rosbag.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore
from typeguard import typechecked
from typing import Tuple, Union
import tqdm

class ImageData(Data):

    # Define image encodings enum
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
            


    # Define image-specific data attributes
    height: int
    width: int
    encoding: ImageEncoding
    images: np.ndarray

    @typechecked
    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], 
                 height: int, width: int, encoding: ImageData.ImageEncoding, images: np.ndarray):
        
        # Copy initial values into attributes
        super().__init__(frame_id, timestamps)
        self.height = height
        self.width = width
        self.encoding = encoding
        self.images = images

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    @typechecked
    def from_ros2_bag(cls, bag_path: Union[Path, str], img_topic: str, save_folder: Union[Path, str]):
        """
        Creates a class structure from a ROS2 bag file with an Image topic. Will
        Also save all the data into .npy and .txt files as this is required if image
        data doesn't fit into the RAM.

        Args:
            bag_path (Path | str): Path to the ROS2 bag file.
            img_topic (str): Topic of the Image messages.
            save_folder (Path | str): Path to save class data into.
        Returns:
            ImageData: Instance of this class.
        """

        # Get topic message count and typestore
        bag_wrapper = Ros2BagWrapper(bag_path, None)
        typestore: Typestore = bag_wrapper.get_typestore()
        num_msgs: int = bag_wrapper.get_topic_count(img_topic)

        # Extract relevant image parameters
        image_shape, frame_id, height, width, encoding = None, None, None, None, None
        with Reader2(bag_path) as reader:
            connections = [x for x in reader.connections if x.topic == img_topic]
            for conn, _, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                frame_id = msg.header.frame_id
                height = msg.height
                width = msg.width
                encoding = ImageData.ImageEncoding.from_ros_str(msg.encoding)
                img = ImageData._decode_image_msg(msg, encoding, height, width)
                image_shape = img.shape
                break
        
        # Pre-allocate arrays (memory-mapped or otherwise)
        imgs_path = str(Path(save_folder) / "imgs.npy")
        os.makedirs(save_folder, exist_ok=True)
        img_memmap = open_memmap(imgs_path, dtype=img.dtype, shape=(num_msgs, *image_shape), mode='w+')
        timestamps_np = np.zeros(num_msgs, dtype=np.float128)

        # Setup tqdm bar
        pbar = tqdm.tqdm(total=num_msgs, desc="Extracting Images...", unit=" msgs")

        # Extract the images/timestamps and save
        with Reader2(bag_path) as reader: 
            i = 0
            connections = [x for x in reader.connections if x.topic == img_topic]
            for conn, _, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

                # Extract images (skipping malformed ones)
                img = None
                try:
                    img = ImageData._decode_image_msg(msg, encoding, height, width)
                except Exception as e:
                    print("Failure decoding image msg: ", e)
                if img is not None and img.shape == image_shape: 
                    img_memmap[i] = img

                # Extract timestamps
                ts = Ros2BagWrapper.extract_timestamp(msg)
                timestamps_np[i] = ts

                # Update the count
                i += 1
                pbar.update(1)

        # Write all images to disk and save timestamps and other data
        img_memmap.flush()
        np.save(str(Path(save_folder) / "times.npy"), timestamps_np, allow_pickle=False)
        with open(str(Path(save_folder) / "attributes.txt"), "w") as f:
            f.write(f"image_shape: {image_shape}\n")
            f.write(f"frame_id: {frame_id}\n")
            f.write(f"height: {height}\n")
            f.write(f"width: {width}\n")
            f.write(f"encoding: {encoding}\n")

        # Create an ImageData class
        return cls(frame_id, timestamps_np, height, width, encoding, np.load(imgs_path, mmap_mode='r+'))
    
    @classmethod
    @typechecked
    def from_npy(cls, folder_path: Union[Path, str]):
        """
        Creates a class structure from .npy and .txt files (the ones written by from_ros2_bag()).

        Args:
            folder_path (Path | str): Path to the folder with:
                - imgs.npy
                - times.npy
                - attributes.txt
        Returns:
            ImageData: Instance of this class.
        """

        # Calculate other paths from folder path
        imgs_path = str(Path(folder_path) / "imgs.npy")
        ts_path = str(Path(folder_path) / "times.npy")
        attr_path = str(Path(folder_path) / "attributes.txt")

        # Read in the attributes
        attr_data = {}
        with open(attr_path, "r") as f:
            for line in f:
                key, val = line.strip().split(":", 1)
                attr_data[key.strip()] = val.strip()

        # Parse and assign values to variables
        frame_id = attr_data["frame_id"]
        height = int(attr_data["height"])
        width = int(attr_data["width"])
        encoding = ImageData.ImageEncoding.from_str(attr_data["encoding"])

        # Create an ImageData class
        return cls(frame_id, np.load(ts_path), height, width, encoding, np.load(imgs_path, mmap_mode='r+'))

    @classmethod
    @typechecked
    def from_npy_files(cls, npy_folder_path: Union[Path, str], frame_id: str):
        """
        Creates a class structure from .npy files, where each individual image
        is stored in an .npy file with the timestamp as the name

        Args:
            npy_folder_path (Path | str): Path to the folder with the npy images.
            frame_id (str): The frame where this image data was collected.
        Returns:
            ImageData: Instance of this class.
        """

        # Get all png files in the designated folder (sorted)
        all_image_files = [str(p) for p in Path(npy_folder_path).glob("*.npy")]

        # Extract the timestamps and sort them
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_image_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]

        # Use sorted_indices to sort all_image_files in the same way
        all_image_files_sorted = [all_image_files[i] for i in sorted_indices]

        # Extract width, height, and channels
        first_image = np.load(all_image_files_sorted[0], 'r')
        assert len(first_image.shape) >= 2
        assert len(first_image.shape) < 4
        height = first_image.shape[0]
        width = first_image.shape[1]
        channels = 1
        if len(first_image.shape) > 2: 
            channels = first_image.shape[2]

        # Extract mode and make sure it matches the supported type for this operation
        encoding = ImageData.ImageEncoding.from_dtype_and_channels(first_image.dtype, channels)
        if encoding != ImageData.ImageEncoding._32FC1:
            raise NotImplementedError(f"Only ImageData.ImageEncoding._32FC1 mode implemented for 'from_npy_files', not {encoding}")
        
        # Load the images as numpy arrays
        assert channels == 1
        images = np.zeros((len(all_image_files_sorted), height, width), dtype=np.float32)
        pbar = tqdm.tqdm(total=len(all_image_files_sorted), desc="Extracting NPY...", unit=" images")
        for i, path in enumerate(all_image_files_sorted):
            images[i] = np.load(path, 'r')
            pbar.update()

        # Return an ImageData class
        return cls(frame_id, timestamps_sorted, height, width, encoding, images)

    @classmethod
    @typechecked
    def from_image_files(cls, image_folder_path: Union[Path, str], frame_id: str):
        """
        Creates a class structure from a folder with .png files, using the file names
        as the timestamps. This is the format that the HERCULES v1.4 dataset provides
        for image data.

        Args:
            image_folder_path (Path | str): Path to the folder with the images.
            frame_id (str): The frame where this image data was collected.
        Returns:
            ImageData: Instance of this class.
        """

        # Get all png files in the designated folder (sorted)
        all_image_files = [str(p) for p in Path(image_folder_path).glob("*.png")]

        # Extract the timestamps and sort them
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_image_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]

        # Use sorted_indices to sort all_image_files in the same way
        all_image_files_sorted = [all_image_files[i] for i in sorted_indices]

        # Make sure the mode is what we expect
        with Image.open(all_image_files_sorted[0]) as first_image:
            encoding = ImageData.ImageEncoding.from_pillow_str(first_image.mode)
            if encoding != ImageData.ImageEncoding.RGB8 and encoding != ImageData.ImageEncoding.Mono8:
                raise NotImplementedError(f"Only RGB8 & Mono8 suppported for 'from_image_files', not \
                                        {encoding}")
        
        # Get dtype and channels based on the encoding
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(encoding)

        # Define the image array shape
        if channels == 1:
            img_arr_shape = (len(all_image_files_sorted), first_image.height, first_image.width)
        else: 
            img_arr_shape = (len(all_image_files_sorted), first_image.height, first_image.width, channels)

        # Load the images as numpy arrays
        images = np.zeros(img_arr_shape, dtype=dtype)
        pbar = tqdm.tqdm(total=len(all_image_files_sorted), desc="Extracting Images...", unit=" images")
        for i, path in enumerate(all_image_files_sorted):
            images[i] = np.array(Image.open(path), dtype=dtype)
            pbar.update()

        # Return an ImageData class
        return cls(frame_id, timestamps_sorted, first_image.height, first_image.width, encoding, images)
    
    # =========================================================================
    # ========================= Manipulation Methods ========================== 
    # =========================================================================  

    @typechecked
    def downscale_by_factor(self, scale: int):
        """
        Scales down all images by the provided factor.

        Args:
            scale (int): The downscaling factor. Must evenly divide both height and width.
        """

        if self.height % scale != 0 or self.width % scale != 0:
            raise ValueError(f"Scale factor {scale} must evenly divide both height ({self.height}) and width ({self.width})")
        
        # Calculate new height/width
        self.height = self.height // scale
        self.width = self.width // scale

        # Ensure we're working with Mono8 data
        if self.encoding != ImageData.ImageEncoding.Mono8:
            raise NotImplementedError(f"This method is only currently implemented for Mono8 data, not {self.encoding}!")

        # Determine the number of channels in the image
        if len(self.images.shape) == 4: channels = self.images.shape[3]
        else: channels = 1

        # Create a new array to hold the resized images
        if channels == 1:
            rescaled_images = np.zeros((self.len(), self.height, self.width), dtype=self.images.dtype)
        else:
            rescaled_images = np.zeros((self.len(), self.height, self.width, channels), dtype=self.images.dtype)
        
        # Resize each image
        for i in range(self.len()):
            rescaled_images[i] = cv2.resize(self.images[i], (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self.images = rescaled_images

    def crop_data(self, start: Decimal, end: Decimal):
        """ Will crop the data so only values within [start, end] inclusive are kept. """

        # Create boolean mask of data to keep
        mask = (self.timestamps >= start) & (self.timestamps <= end)

        # Apply mask
        self.timestamps = self.timestamps[mask]
        self.images = self.images[mask]

    # =========================================================================
    # ============================ Export Methods ============================= 
    # ========================================================================= 

    @typechecked
    def to_npy(self, output_folder_path: Union[Path, str]):
        """
        Saves each image in this ImageData into three files:
        - imgs.npy (with image data)
        - times.npy (with timestamps)
        - attributes.txt

        Args:
            output_folder_path (Path | str): The folder to save the .npy file at.
        """

        # Setup the output directory
        output_path = Path(output_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check that the encoding is supported
        if self.encoding != ImageData.ImageEncoding.RGB8 and self.encoding != ImageData.ImageEncoding._32FC1:
            raise NotImplementedError(f"Only RGB8 & 32FC1 images have been tested for export, not {self.encoding}")

        # Get dtype and channels
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(self.encoding)

        # Save images into memory-mapped array
        shape = (self.len(), self.height, self.width) if channels == 1 else (self.len(), self.height, self.width, channels)
        img_memmap = open_memmap(str(Path(output_folder_path) / "imgs.npy"), dtype=dtype, shape=shape, mode='w+')
        pbar = tqdm.tqdm(total=self.len(), desc="Saving Images...", unit=" images")
        for i in range(self.len()):
            img_memmap[i] = self.images[i]
            pbar.update()
        img_memmap.flush()

        # Save the timestamps
        np.save(str(Path(output_folder_path) / "times.npy"), self.timestamps.astype(np.float128), allow_pickle=False)

        # Save attributes
        with open(str(Path(output_folder_path) / "attributes.txt"), "w") as f:
            f.write(f"image_shape: ({self.height},{self.width})\n")
            f.write(f"frame_id: {self.frame_id}\n")
            f.write(f"height: {self.height}\n")
            f.write(f"width: {self.width}\n")
            f.write(f"encoding: {self.encoding}\n")


    @typechecked
    def to_image_files(self, output_folder_path: Union[Path, str]):
        """
        Saves each image in this ImageData instance to the specified folder,
        using the timestamps as filenames in .png format (lossless compression).

        Args:
            output_folder_path (Path | str): The folder to save images into.
        """

        # Setup the output directory
        output_path = Path(output_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check that the encoding is Mono8
        if self.encoding != ImageData.ImageEncoding.Mono8:
            raise NotImplementedError(f"Only Mono8 encoding currently supported for export, not {self.encoding}")

        # Setup a progress bar
        pbar = tqdm.tqdm(total=self.images.shape[0], desc="Saving Images...", unit=" images")

        # Save each image
        for i, timestamp in enumerate(self.timestamps):
            # Format timestamp to match input expectations
            filename = f"{timestamp:.9f}" + ".png"
            file_path = output_path / filename

            # Save as lossless PNG with default compression
            img = Image.fromarray(self.images[i], mode="L")
            img.save(file_path, format="PNG", compress_level=1)
            pbar.update()

        pbar.close()

    # =========================================================================
    # ============================ Image Decoding ============================= 
    # ========================================================================= 

    @staticmethod
    @typechecked
    def _decode_image_msg(msg: object, encoding: ImageData.ImageEncoding, height: int, width: int):
        """
        Helper method that decodes image data from a ROS2 Image message.

        Args:
            msg (object): The ROS2 Image message.
            encoding (ImageEncoding): The encoding of the image data.
            height (int): Height of the image.
            width (int): Width of the image .
        """
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(encoding)
        if channels > 1:
            return np.frombuffer(msg.data, dtype=dtype).reshape((height, width, channels)) 
        else:
            return np.frombuffer(msg.data, dtype=dtype).reshape((height, width))
        
    # =========================================================================
    # ======================= Multi ImageData Methods ========================= 
    # ========================================================================= 

    def compare_timestamps(self, other: ImageData):
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

    @typechecked
    def stereo_undistort_and_rectify(self: ImageData, other: ImageData,
            K1: np.ndarray, D1: np.ndarray, K2: np.ndarray, D2: np.ndarray, 
            R: np.ndarray, T: np.ndarray) -> Tuple[ImageData, ImageData, np.ndarray, np.ndarray]:
        """
        Undistort and rectify stereo images using stereo calibration parameters. 
        Note that self NEEDS to be the left stereo image sequence.

        Args:
            other (ImageData): The right stereo image sequence.
            K1, D1: Intrinsics and distortion for left camera.
            K2, D2: Intrinsics and distortion for right camera.
            R, T: Rotation and translation from left to right camera.

        Returns:
            Tuple[ImageData, ImageData, np.ndarray, np.ndarray]: Rectified left and right 
                ImageData, and new Instrinsics matrices for the left and right cameras.
        """

        # Make sure the ImageData sequences are compatible
        assert self.width == other.width and self.height == other.height and self.encoding == other.encoding, \
            "Left and right images must have the same resolution and encoding."

        # Find matching timestamps between self and other
        set_self = set(self.timestamps)
        set_other = set(other.timestamps)
        common_timestamps = sorted(set_self.intersection(set_other))
        if len(common_timestamps) == 0:
            raise ValueError("No matching timestamps between left and right images.")
        
        # Find indices of matching timestamps in each ImageData
        left_indices = [np.where(self.timestamps == ts)[0][0] for ts in common_timestamps]
        right_indices = [np.where(other.timestamps == ts)[0][0] for ts in common_timestamps]

        # Image size
        image_size = (self.width, self.height)

        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        # Compute intrinsics of rectified imagery
        K1_new = P1[:, :3]
        K2_new = P2[:, :3]
        print("New left camera intrinsics after rectification:\n",  K1_new)
        print("New right camera intrinsics after rectification:\n", K2_new)
        print("Distortion coefficients after rectification are zero.")

        # Compute rectification maps
        map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

        # Allocate arrays for rectified images (only matching pairs)
        left_rectified = np.zeros((len(common_timestamps), self.height, self.width, *self.images.shape[3:]), dtype=self.images.dtype)
        right_rectified = np.zeros((len(common_timestamps), other.height, other.width, *other.images.shape[3:]), dtype=other.images.dtype)

        # Rectify/Undistort each image pair
        for i, (li, ri) in enumerate(tqdm.tqdm(zip(left_indices, right_indices), total=len(common_timestamps), desc="Rectifying stereo pairs")):
            left_rectified[i] = cv2.remap(self.images[li], map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
            right_rectified[i] = cv2.remap(other.images[ri], map2_x, map2_y, interpolation=cv2.INTER_LINEAR)

        # Return new ImageData instances with rectified images and matched timestamps
        left = ImageData(self.frame_id, np.array(common_timestamps), self.height, self.width, self.encoding, left_rectified)
        right = ImageData(other.frame_id, np.array(common_timestamps), other.height, other.width, other.encoding, right_rectified)
        return left, right, K1_new, K2_new


    # =========================================================================
    # =========================== Conversion to ROS =========================== 
    # ========================================================================= 

    @typechecked
    @staticmethod
    def get_ros_msg_type() -> str:
        """ Return the __msgtype__ for an Image msg. """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        return typestore.types['sensor_msgs/msg/Image'].__msgtype__

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
        Image, Header, Time = typestore.types['sensor_msgs/msg/Image'], typestore.types['std_msgs/msg/Header'], typestore.types['builtin_interfaces/msg/Time']

        # Calculate the step based on encoding
        if self.encoding == ImageData.ImageEncoding.RGB8:
            step = 3 * self.width
        elif self.encoding == ImageData.ImageEncoding._32FC1:
            step = 4 * self.width
        elif self.encoding == ImageData.ImageEncoding.Mono8:
            step = self.width
        else:
            raise ValueError(f"Unsupported encoding: {self.encoding}")

        # Get the seconds and nanoseconds
        seconds = int(self.timestamps[i])
        nanoseconds = (self.timestamps[i] - self.timestamps[i].to_integral_value(rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN)

        # Calculate the ROS2 Image data
        if self.encoding == ImageData.ImageEncoding.RGB8:
            data = self.images[i].flatten()
        elif self.encoding == ImageData.ImageEncoding._32FC1:
            data = self.images[i].flatten().view(np.uint8)

        # Write the data into the new class
        return Image(Header(stamp=Time(sec=int(seconds), 
                                    nanosec=int(nanoseconds)), 
                            frame_id=self.frame_id),
                    height=self.height, 
                    width=self.width, 
                    encoding=ImageData.ImageEncoding.to_ros_str(self.encoding),
                    is_bigendian=0, 
                    step=step, 
                    data=data)
        

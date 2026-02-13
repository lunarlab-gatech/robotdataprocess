from __future__ import annotations

from ...conversion_utils import col_to_dec_arr
import cv2

from decimal import Decimal
from enum import Enum
from .ImageData import ImageData
import numpy as np
from numpy.lib.format import open_memmap
import os
from pathlib import Path
from PIL import Image
from ...ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys.store import Typestore
from typeguard import typechecked
from typing import Tuple, Union
import tqdm

@typechecked
class ImageDataInMemory(ImageData):
    """
    Image data stored entirely in RAM (or as a memory-mapped numpy array).

    Supports loading from ROS2 bags, ``.npy`` files, and ``.png`` folders.
    Provides in-memory downscaling and export to PNG or ``.npy`` format.
    """

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], 
                 height: int, width: int, encoding: ImageData.ImageEncoding, images: np.ndarray):
        super().__init__(frame_id, timestamps, height, width, encoding, images)

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in ImageDataInMemory. """
        pass

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
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
                img = ImageDataInMemory._decode_image_msg(msg, encoding, height, width)
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
                    img = ImageDataInMemory._decode_image_msg(msg, encoding, height, width)
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

        # Get all npy files in the designated folder (sorted)
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
        pbar = tqdm.tqdm(total=len(all_image_files_sorted), desc="Extracting Images...", unit=" images")
        for i, path in enumerate(all_image_files_sorted):
            images[i] = np.load(path, 'r')
            pbar.update()

        # Return an ImageData class
        return cls(frame_id, timestamps_sorted, height, width, encoding, images)

    @classmethod
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

    # =========================================================================
    # ============================ Export Methods ============================= 
    # ========================================================================= 

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
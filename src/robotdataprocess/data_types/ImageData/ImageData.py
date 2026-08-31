from __future__ import annotations

from ..Data import ROSMsgLibType
from ..SequentialData import SequentialData
import cv2
import decimal
from decimal import Decimal
from enum import Enum
from ...utils.math_utils import nearest_index
from ...utils.ModuleImporter import ModuleImporter
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from PIL import Image
from typeguard import typechecked
from typing import Union, Any, Optional, Tuple
import tqdm
from rosbags.typesys import Stores, get_typestore
from ...utils.VideoGenerator import VideoGenerator

@typechecked
class ImageData(SequentialData):
    """ Generic ImageData class that should be overwritten by children """

    # Define image encodings enumeration
    class ImageEncoding(Enum):
        Mono8 = 0
        RGB8 = 1
        _32FC1 = 2
        BGR8 = 3
        _16UC1 = 4

        # ================ Class Methods ================
        @classmethod
        def from_str(cls, encoding_str: str):
            if encoding_str == "ImageEncoding.Mono8":
                return cls.Mono8
            elif encoding_str == "ImageEncoding.RGB8":
                return cls.RGB8
            elif encoding_str == "ImageEncoding._32FC1":
                return cls._32FC1
            elif encoding_str == "ImageEncoding.BGR8":
                return cls.BGR8
            elif encoding_str == "ImageEncoding._16UC1":
                return cls._16UC1
            else:
                raise NotImplementedError(f"This encoding ({encoding_str}) is not yet implemented (or it doesn't exist)!")

        @classmethod
        def from_ros2_str(cls, encoding_str: str):
            encoding_str = encoding_str.lower()
            if encoding_str == 'mono8':
                return cls.Mono8
            elif encoding_str == 'rgb8':
                return cls.RGB8
            elif encoding_str == "32fc1":
                return cls._32FC1
            elif encoding_str == 'bgr8':
                return cls.BGR8
            elif encoding_str == '16uc1':
                return cls._16UC1
            else:
                raise NotImplementedError(f"This encoding ({encoding_str}) is not yet implemented (or it doesn't exist)!")

        @classmethod
        def from_dtype_and_channels(cls, dtype: np.dtype, channels: int):
            if dtype == np.uint8 and channels == 1:
                return cls.Mono8
            elif dtype == np.uint8 and channels == 3:
                raise NotImplementedError(f"dtype {dtype} w/ {channels} channel(s) can't determine which encoding the data is in!")
            elif dtype == np.float32 and channels == 1:
                return cls._32FC1
            elif dtype == np.uint16 and channels == 1:
                return cls._16UC1
            else:
                raise NotImplementedError(f"dtype {dtype} w/ {channels} channel(s) has no corresponding encoding!")
        
        @classmethod
        def from_compressed_ros1_str(cls, format_str: str):
            """
            Extract encoding from a ROS1 CompressedImage ``format`` field.

            The field follows the convention
            ``"<original_enc>; <codec> compressed [<stored_enc>]"``.
            When a stored encoding is present (e.g. ``"rgb8; jpeg compressed bgr8"``)
            it is used directly; when absent (e.g. ``"mono8; png compressed"``) the
            original encoding before the semicolon is used.

            Args:
                format_str: The ``format`` field of a ``sensor_msgs/msg/CompressedImage``.
            Returns:
                ImageEncoding matching the encoding of the decompressed image data.
            Raises:
                NotImplementedError: If the encoding token is not recognised.
                ValueError: If the format string cannot be parsed.
            """
            lower = format_str.lower().strip()
            if 'compressed' in lower:
                after = lower.split('compressed', 1)[1].strip()
                if after:
                    return cls.from_ros2_str(after)
                before = lower.split(';')[0].strip()
                return cls.from_ros2_str(before)
            raise ValueError(
                f"Cannot determine encoding from compressed format string: {format_str!r}")

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
        def to_ros2_str(encoding: ImageData.ImageEncoding):
            if encoding == ImageData.ImageEncoding.Mono8:
                return 'mono8'
            elif encoding == ImageData.ImageEncoding.RGB8:
                return 'rgb8'
            elif encoding == ImageData.ImageEncoding._32FC1:
                return '32FC1'
            elif encoding == ImageData.ImageEncoding.BGR8:
                return 'bgr8'
            elif encoding == ImageData.ImageEncoding._16UC1:
                return '16UC1'
            else:
                raise NotImplementedError(f"This ImageData.ImageEncoding.{encoding} is not yet implemented (or it doesn't exist)!")
        
        @staticmethod
        def to_dtype_and_channels(encoding):
            if encoding == ImageData.ImageEncoding.Mono8:
                return (np.uint8, 1)
            elif encoding == ImageData.ImageEncoding.RGB8:
                return (np.uint8, 3)
            elif encoding == ImageData.ImageEncoding.BGR8:
                return (np.uint8, 3)
            elif encoding == ImageData.ImageEncoding._32FC1:
                return (np.float32, 1)
            elif encoding == ImageData.ImageEncoding._16UC1:
                return (np.uint16, 1)
            else:
                raise NotImplementedError(f"This encoding ({encoding}) is missing a mapping to dtype/channels!")

        # ================ Conversion Methods ================
        @staticmethod
        def get_encoding_conversion(from_encoding: 'ImageData.ImageEncoding', to_encoding: 'ImageData.ImageEncoding'):
            """
            Returns a function that converts a single image array from from_encoding to
            to_encoding. Currently supports RGB8 -> BGR8, Mono8 -> BGR8, and Mono8 -> RGB8.

            Args:
                from_encoding: The image's current encoding.
                to_encoding: The encoding to convert the image to.
            Returns:
                Callable[[np.ndarray], np.ndarray]: Converts an image array from from_encoding to to_encoding.
            Raises:
                NotImplementedError: If the conversion between the two encodings is not supported.
            """
            if from_encoding == to_encoding:
                return lambda image: image
            elif from_encoding == ImageData.ImageEncoding.RGB8 and to_encoding == ImageData.ImageEncoding.BGR8:
                return lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif from_encoding == ImageData.ImageEncoding.Mono8 and to_encoding == ImageData.ImageEncoding.BGR8:
                return lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif from_encoding == ImageData.ImageEncoding.Mono8 and to_encoding == ImageData.ImageEncoding.RGB8:
                return lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                raise NotImplementedError(f"Encoding conversion from {from_encoding} to {to_encoding} is not supported.")

    height: int
    width: int
    encoding: ImageData.ImageEncoding
    images: Union[np.ndarray, Any] # With Any being a LazyImageArray

    _COMPRESSED_MSGTYPE: str = 'sensor_msgs/msg/CompressedImage'

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], height: int,
                 width: int, encoding: ImageData.ImageEncoding, images: Union[np.ndarray, Any]):
        super().__init__(frame_id, timestamps)
        self.height = height
        self.width = width
        self.encoding = encoding
        self.images = images

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in ImageData. """
        pass

    def __eq__(self, other) -> bool:
        parent_result = super().__eq__(other)
        if parent_result is not True:
            return parent_result
        if self.height != other.height or self.width != other.width:
            print(f"  [__eq__] height/width: ({self.height}, {self.width}) != ({other.height}, {other.width})")
            return False
        if self.encoding != other.encoding:
            print(f"  [__eq__] encoding: {self.encoding} != {other.encoding}")
            return False
        if len(self.images) != len(other.images):
            print(f"  [__eq__] images length: {len(self.images)} != {len(other.images)}")
            return False
        # Compare frame by frame, since `images` may be an in-memory array or a lazily
        # loaded on-disk array -- indexing works uniformly for both, but is only cheap for
        # the former.
        for i in range(len(self.images)):
            if not np.array_equal(self.images[i], other.images[i]):
                print(f"  [__eq__] images first diff at idx {i}")
                return False
        return True

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @staticmethod
    def decode_image_msg(msg: Any, encoding: Optional['ImageData.ImageEncoding'] = None,
                          height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
        """
        Decodes raw pixel data from a ROS Image message into a contiguous numpy array,
        honouring ``msg.step`` (row padding) and ``msg.is_bigendian`` when present.

        Args:
            msg: A ROS Image message with a ``data`` field, and (unless passed explicitly)
                ``encoding``, ``height`` and ``width`` fields. ``step`` and ``is_bigendian``
                are read when present and otherwise assumed to be unpadded/little-endian.
            encoding: The image encoding to interpret the bytes as; None reads ``msg.encoding``.
            height: Image height in pixels; None reads ``msg.height``.
            width: Image width in pixels; None reads ``msg.width``.
        Returns:
            np.ndarray: Decoded image array. Shape is (H, W, C) for multi-channel
                encodings or (H, W) for single-channel.
        Raises:
            ValueError: If ``msg.step`` is not a whole number of pixels wide, or is too
                narrow to hold one row of ``width * channels`` pixels.
        """
        # Load in parameters if passed
        if encoding is None:
            encoding = ImageData.ImageEncoding.from_ros2_str(msg.encoding)
        if height is None:
            height = msg.height
        if width is None:
            width = msg.width

        # Sanity check values
        dtype, channels = ImageData.ImageEncoding.to_dtype_and_channels(encoding)
        itemsize: int = np.dtype(dtype).itemsize
        row_pixels: int = width * channels
        step: int = getattr(msg, 'step', row_pixels * itemsize)
        if step % itemsize != 0:
            raise ValueError(f"Image step {step} is not a multiple of the {dtype} itemsize {itemsize}.")
        if step // itemsize < row_pixels:
            raise ValueError(f"Image step {step} is too narrow to hold one row of {row_pixels} "
                              f"pixels ({itemsize} bytes each).")

        # Get dtype to handle endianness
        read_dtype: np.dtype = np.dtype(dtype).newbyteorder('>') if getattr(msg, 'is_bigendian', 0) else np.dtype(dtype)

        # Decode the image data handling padding if necessary
        rows: np.ndarray = np.frombuffer(msg.data, dtype=read_dtype).reshape(height, step // itemsize)
        image: np.ndarray = rows[:, :row_pixels]
        if channels > 1:
            image = image.reshape(height, width, channels)
        return np.ascontiguousarray(image, dtype=dtype)

    @staticmethod
    def _decode_compressed_image_msg(msg: Any) -> Tuple[np.ndarray, 'ImageData.ImageEncoding']:
        """
        Decompresses a sensor_msgs/msg/CompressedImage message into a numpy array.

        Parses the encoding from ``msg.format`` via
        ``ImageEncoding.from_compressed_ros1_str``, decompresses the payload with
        ``cv2.imdecode``, then calls ``decode_image_msg`` for consistent
        dtype/shape handling.

        Args:
            msg: A ROS1 CompressedImage message with ``format`` and ``data`` fields.
        Returns:
            Tuple of (image array, encoding).
        """
        decoded = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError("cv2.imdecode failed to decode compressed image data.")
        height, width = decoded.shape[:2]

        try:
            encoding = ImageData.ImageEncoding.from_compressed_ros1_str(msg.format)
        except ValueError:
            # Simple format string (e.g. 'jpg', 'png') with no encoding prefix — infer from shape.
            # cv2.imdecode always produces BGR for color images.
            channels = 1 if decoded.ndim == 2 else decoded.shape[2]
            if channels == 3:
                encoding = ImageData.ImageEncoding.BGR8
            elif channels == 1:
                encoding = ImageData.ImageEncoding.Mono8
            else:
                raise NotImplementedError(
                    f"Cannot infer encoding for compressed image with {channels} channels "
                    f"and format string {msg.format!r}.")

        class _RawProxy:
            data = decoded.tobytes()

        return ImageData.decode_image_msg(_RawProxy(), encoding, height, width), encoding

    @classmethod
    def from_image_files(cls, image_folder_path: Union[Path, str], frame_id: str) -> ImageData:
        """
        Creates a class structure from a folder with .png files.

        Args:
            image_folder_path: Path to the folder containing image files.
            frame_id: The frame ID to assign.

        Returns:
            ImageData: Instance of this class.
        """
        NotImplementedError("This method needs to be overwritten by the child Data class!")
    
    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    @staticmethod
    def convert_image_encoding(image: np.ndarray, from_encoding: 'ImageData.ImageEncoding',
                                to_encoding: 'ImageData.ImageEncoding') -> np.ndarray:
        """
        Converts a single decoded image array from from_encoding to to_encoding.

        Args:
            image: The image array, in from_encoding.
            from_encoding: The image's current encoding.
            to_encoding: The encoding to convert the image to.
        Returns:
            np.ndarray: A contiguous array holding the image in to_encoding.
        Raises:
            NotImplementedError: If the conversion between the two encodings is not supported.
        """
        convert = ImageData.ImageEncoding.get_encoding_conversion(from_encoding, to_encoding)
        return np.ascontiguousarray(convert(image))

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
        self.images = self.images[mask]

    # =========================================================================
    # ============================ Export Methods ============================= 
    # ========================================================================= 

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
        if self.encoding != ImageData.ImageEncoding.RGB8 and self.encoding != ImageData.ImageEncoding._32FC1 \
                and self.encoding != ImageData.ImageEncoding.Mono8:
            raise NotImplementedError(f"Only RGB8, Mono8 & 32FC1 images have been tested for export, not {self.encoding}")

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
        if self.encoding != ImageData.ImageEncoding.Mono8 and self.encoding != ImageData.ImageEncoding.RGB8:
            raise NotImplementedError(f"Only Mono8 and RGB8 encoding currently supported for export, not {self.encoding}")

        # Setup a progress bar
        pbar = tqdm.tqdm(total=self.len(), desc="Saving Images...", unit=" images")

        # Save each image
        for i, timestamp in enumerate(self.timestamps):
            # Format timestamp to match input expectations
            filename = f"{timestamp:.9f}" + ".png"
            file_path = output_path / filename

            # Determine mode
            if self.encoding == ImageData.ImageEncoding.Mono8:
                mode = "L"
            elif self.encoding == ImageData.ImageEncoding.RGB8:
                mode = "RGB"
            else:
                raise Exception("Should have been caught already!")

            # Save as lossless PNG with default compression
            img = Image.fromarray(self.images[i], mode=mode)
            img.save(file_path, format="PNG", compress_level=1)
            pbar.update()

        pbar.close()

    def to_mp4(self, output_path: Union[Path, str], fps: float, video_duration_sec: float,
               max_frame_time_margin_sec: float = 0.1):
        """
        Saves the images in this ImageData instance as an .mp4 video, resampled
        to fps and video_duration_sec instead of writing every source image --
        e.g. pass a video_duration_sec shorter than the source's real-time
        span for a faster-than-real-time video, or longer for a slower one.

        Source timestamps are linearly scaled into [0, video_duration_sec],
        and only round(fps * video_duration_sec) frames are written, each the
        nearest-in-time source image to its evenly-spaced output sample. This
        avoids writing (and immediately throwing away at playback) every
        source frame just to speed up playback.

        Args:
            output_path (Path | str): The file path to save the video to.
            fps (float): The output video's frame rate.
            video_duration_sec (float): Desired total video duration.
            max_frame_time_margin_sec (float): Maximum allowed gap, in the
                scaled video_duration_sec timeline, between an output sample
                and its nearest source frame before raising -- catches
                genuine gaps in the source timestamps rather than expected
                sampling sparsity.

        Raises:
            NotImplementedError: If self.encoding isn't Mono8, RGB8, or BGR8.
            ValueError: If self has fewer than 2 timestamps, or some output
                sample has no source frame within max_frame_time_margin_sec.
        """

        # Check that the encoding is supported
        if self.encoding != ImageData.ImageEncoding.Mono8 and self.encoding != ImageData.ImageEncoding.RGB8 \
                and self.encoding != ImageData.ImageEncoding.BGR8:
            raise NotImplementedError(f"Only Mono8, RGB8 & BGR8 encoding currently supported for to_mp4, not {self.encoding}")
        if self.len() < 2:
            raise ValueError("Cannot resample with fewer than 2 timestamps.")

        # Scale source timestamps into [0, video_duration_sec]
        source_times = self.timestamps.astype(np.float64)
        scale = video_duration_sec / (source_times[-1] - source_times[0])
        scaled_source_times = (source_times - source_times[0]) * scale

        # Build the evenly-spaced output sample times, and find each one's nearest source frame
        num_output_frames = max(int(round(fps * video_duration_sec)), 1)
        output_times = np.linspace(0.0, video_duration_sec, num_output_frames, endpoint=False)
        frame_indices = nearest_index(scaled_source_times, output_times)

        gaps = np.abs(scaled_source_times[frame_indices] - output_times)
        if np.any(gaps > max_frame_time_margin_sec):
            worst = int(np.argmax(gaps))
            raise ValueError(
                f"No source frame within max_frame_time_margin_sec ({max_frame_time_margin_sec:.4f}s) of output "
                f"sample at {output_times[worst]:.4f}s (nearest is {gaps[worst]:.4f}s away).")

        # Write each selected image as a video frame (converting it to BGR8 first)
        conversion = ImageData.ImageEncoding.get_encoding_conversion(self.encoding, ImageData.ImageEncoding.BGR8)
        pbar = tqdm.tqdm(total=len(frame_indices), desc="Writing Video...", unit=" frames")
        try:
            with VideoGenerator.open_video_writer(output_path, fps, (self.width, self.height)) as writer:
                for i in frame_indices:
                    writer.write(conversion(self.images[i]))
                    pbar.update()
        finally:
            pbar.close()

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # ========================================================================= 

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """
        Return the __msgtype__ for an Image msg.

        Args:
            lib_type: The ROS message library to use.

        Returns:
            The Image message type for the specified library.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

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
                        encoding=ImageData.ImageEncoding.to_ros2_str(self.encoding),
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
            img_msg.encoding = ImageData.ImageEncoding.to_ros2_str(self.encoding)
            img_msg.is_bigendian = 0
            img_msg.step = step
            img_msg.data = data.tolist()
            return img_msg

        else:
            raise NotImplementedError(f"Unsupported ROS_MSG_LIBRARY_TYPE {lib_type} for ImageData.get_ros_msg()!")

    # =========================================================================
    # ========================= Multi ImageData Methods =========================
    # =========================================================================

    @staticmethod
    def crop_to_matched(data1: ImageData, data2: ImageData, tolerance: Decimal) -> None:
        """
        Crop two ImageData objects in place so only mutually-matched entries
        remain.

        Args:
            data1: The first ImageData object, cropped in place.
            data2: The second ImageData object, cropped in place.
            tolerance: Maximum allowed absolute time difference between
                matched timestamps.

        Raises:
            NotImplementedError: Always; must be overridden with real logic.
        """
        raise NotImplementedError("This method needs to be overwritten by the child Data class!")
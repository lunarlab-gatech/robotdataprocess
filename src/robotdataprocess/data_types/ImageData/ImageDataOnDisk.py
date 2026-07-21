from __future__ import annotations
from ...conversion_utils import col_to_dec_arr
from ..CameraData import CameraData
from .ImageData import ImageData
import cv2
import copy
from decimal import Decimal
import numpy as np
from pathlib import Path
from PIL import Image
from rosbags.rosbag1 import Reader as Reader1
from rosbags.typesys import Stores, get_typestore
import tqdm
from typeguard import typechecked
from typing import Any, Tuple, Union, List, Callable

class LazyImageArray:
    """A read-only array-like interface that loads PNG images from disk on demand."""
    
    container: 'ImageDataOnDisk'
    image_paths: List[Path]
    transformations: List[Callable]

    def __init__(self, container: 'ImageDataOnDisk', image_paths: List[Path], transformations: Union[List[Callable], None] = None):
        self.container = container
        self.image_paths = image_paths
        self.transformations = copy.deepcopy(transformations) if transformations else []

    def __getitem__(self, idx) -> np.ndarray:
        # Handle boolean masking (used by the crop_data function)
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            new_paths = [p for p, keep in zip(self.image_paths, idx) if keep]
            return self.__class__(self.container, new_paths, self.transformations)

        # Handle slicing (e.g., images[0:10])
        if isinstance(idx, slice):
            return self.__class__(self.container, self.image_paths[idx], self.transformations)

        # Handle single integer indexing (loading the actual image)
        path: Path = Path(self.image_paths[idx])
        if path.suffix == '.npy':
            image = np.load(str(path), 'r')
        elif path.suffix == '.png':
            image = np.array(Image.open(str(path)), dtype=self.dtype)
        else: 
            raise NotImplementedError(f"Unsupported file format {path.suffix} in LazyImageArray!")

        # Apply transformations
        for transform in self.transformations:
            image = transform(image)
        
        return image

    def __setitem__(self, idx, value):
        raise RuntimeError("This LazyPNGArray is read-only; writes are forbidden.")

    def __len__(self):
        return len(self.image_paths)

    @property
    def shape(self):
        _, channels = ImageData.ImageEncoding.to_dtype_and_channels(self.container.encoding)
        if channels == 1:
            return (len(self), self.container.height, self.container.width)
        else:
            return (len(self), self.container.height, self.container.width, 3)

    @property
    def dtype(self):
        dtype_val, _ = ImageData.ImageEncoding.to_dtype_and_channels(self.container.encoding)
        return dtype_val

class BagLazyImageArray:
    """
    A read-only array-like interface that loads images from a ROS1 bag on demand.

    Header-stamp timestamps are held in memory alongside an already-open
    ``Reader1`` instance and a ``header_to_rec_ns`` dict that maps each
    header-stamp (integer nanoseconds) to its bag recording time (integer
    nanoseconds).  On access, the recording time is used with
    ``reader.messages(start=rec_ns, stop=rec_ns+1)`` to seek directly to the
    right position; the retrieved message's header stamp is then verified
    against the target before returning.  The reader is shared across slices
    and boolean-masked views and stays open for the lifetime of the object.
    """

    container: 'ImageDataOnDisk'
    reader: Reader1
    conn_id: int
    timestamps: List[Decimal]
    _header_to_rec_ns: dict  # {header_stamp_ns (int): recording_time_ns (int)}
    msgtype: str
    typestore: Any  # rosbags Typestore
    transformations: List[Callable]

    def __init__(self, container: 'ImageDataOnDisk', reader: Reader1, conn_id: int,
                 timestamps: List[Decimal], header_to_rec_ns: dict, msgtype: str, typestore: Any,
                 transformations: Union[List[Callable], None] = None) -> None:
        self.container = container
        self.reader = reader
        self.conn_id = conn_id
        self.timestamps = timestamps
        self._header_to_rec_ns = header_to_rec_ns
        self.msgtype = msgtype
        self.typestore = typestore
        self.transformations = copy.deepcopy(transformations) if transformations else []

    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> Union[np.ndarray, 'BagLazyImageArray']:
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            new_ts: List[Decimal] = [t for t, keep in zip(self.timestamps, idx) if keep]
            return BagLazyImageArray(self.container, self.reader, self.conn_id,
                                     new_ts, self._header_to_rec_ns, self.msgtype,
                                     self.typestore, self.transformations)

        if isinstance(idx, slice):
            return BagLazyImageArray(self.container, self.reader, self.conn_id,
                                     self.timestamps[idx], self._header_to_rec_ns, self.msgtype,
                                     self.typestore, self.transformations)

        target_ns: int = int(self.timestamps[idx] * Decimal('1e9'))
        rec_ns: int = self._header_to_rec_ns[target_ns]
        conns = [c for c in self.reader.connections if c.id == self.conn_id]
        for _, _, rawdata in self.reader.messages(connections=conns, start=rec_ns, stop=rec_ns + 1):
            msg = self.typestore.deserialize_ros1(rawdata, self.msgtype)
            if msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec != target_ns:
                continue
            if self.msgtype == ImageData._COMPRESSED_MSGTYPE:
                image, _ = ImageData._decode_compressed_image_msg(msg)
            else:
                image = ImageData._decode_image_msg(
                    msg, self.container.encoding, msg.height, msg.width).copy()
            for transform in self.transformations:
                image = transform(image)
            return image

        raise RuntimeError(f"No message at timestamp {self.timestamps[idx]} found in bag.")

    def __setitem__(self, idx: Any, value: Any) -> None:
        raise RuntimeError("This BagLazyImageArray is read-only; writes are forbidden.")

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def shape(self) -> Tuple[int, ...]:
        _, channels = ImageData.ImageEncoding.to_dtype_and_channels(self.container.encoding)
        if channels == 1:
            return (len(self), self.container.height, self.container.width)
        return (len(self), self.container.height, self.container.width, channels)

    @property
    def dtype(self) -> np.dtype:
        dtype_val, _ = ImageData.ImageEncoding.to_dtype_and_channels(self.container.encoding)
        return dtype_val


@typechecked
class ImageDataOnDisk(ImageData):
    """
    Image data loaded lazily from disk, reading each image only when accessed.
    Uses a ``LazyImageArray`` that loads PNG or ``.npy`` files on demand,
    keeping memory usage low for large image sequences. Supports loading from
    ``.png`` and ``.npy`` folders where filenames encode timestamps.
    """
        
    images: LazyImageArray # Not initalized here, but put here for visual code highlighting
    _bag_reader: Union[Reader1, None]

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], height: int, width: int,
                 encoding: ImageData.ImageEncoding, image_paths: List[Path],
                 transformations: Union[List[Callable], None] = None,
                 bag_reader: Union[Reader1, None] = None):

        super().__init__(frame_id, timestamps, height, width, encoding, None)
        transformations_copy = copy.deepcopy(transformations) if transformations else []
        self.images = LazyImageArray(self, image_paths, transformations_copy)
        self._bag_reader = bag_reader

    def __del__(self) -> None:
        if self._bag_reader is not None:
            self._bag_reader.close()

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in ImageDataOnDisk. """
        pass

    # =========================================================================
    # ====================== Encoding Transformations =========================
    # =========================================================================

    def to_encoding(self, encoding: ImageData.ImageEncoding):
        """
        Swap the encoding of the image data.
        Currently only supports RGB8 -> BGR8.

        Args:
            encoding: The target encoding to convert to.

        Raises:
            NotImplementedError: If the conversion between the current and
                target encoding is not supported.
        """
        if encoding == self.encoding:
            return

        if self.encoding == ImageData.ImageEncoding.RGB8 and encoding == ImageData.ImageEncoding.BGR8:
            def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.images.transformations.append(rgb_to_bgr)
            self.encoding = ImageData.ImageEncoding.BGR8
        else:
            raise NotImplementedError(f"Encoding conversion from {self.encoding} to {encoding} is not supported.")

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_image_files(cls, image_folder_path: Union[Path, str], frame_id: str) -> ImageDataOnDisk:
        """
        Creates a class structure from a folder with .png files, using the file names
        as the timestamps.

        Args:
            image_folder_path (Path | str): Path to the folder with the images.
            frame_id (str): The frame where this image data was collected.
        Returns:
            ImageDataOnDisk: Instance of this class.
        """

        # Get all png files in the designated folder (sorted)
        all_image_files = [str(p) for p in Path(image_folder_path).glob("*.png")]

        # Extract the timestamps and sort them
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_image_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]

        # Use sorted_indices to sort all_image_files in the same way
        all_image_files_sorted: List[Path] = [Path(all_image_files[i]) for i in sorted_indices]

        # Make sure the mode is what we expect
        with Image.open(str(all_image_files_sorted[0])) as first_image:
            encoding = ImageData.ImageEncoding.from_pillow_str(first_image.mode)
            if encoding != ImageData.ImageEncoding.RGB8 and encoding != ImageData.ImageEncoding.Mono8:
                raise NotImplementedError(f"Unsupported encoding {encoding} for 'from_image_files' method!")
            height = first_image.height
            width = first_image.width
        
        # Return an ImageDataOnDisk class
        return cls(frame_id, timestamps_sorted, height, width, encoding, all_image_files_sorted)
    
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
        all_image_files_sorted: List[Path] = [Path(all_image_files[i]) for i in sorted_indices]

        # Extract width, height, and channels
        first_image = np.load(str(all_image_files_sorted[0]), 'r')
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

        # Return an ImageDataOnDisk class
        return cls(frame_id, timestamps_sorted, height, width, encoding, all_image_files_sorted)

    @classmethod
    def from_ros1_bag(cls, bag_path: Union[Path, str], img_topic: str) -> 'ImageDataOnDisk':
        """
        Creates a class structure from a ROS1 bag file lazily.  The bag is
        opened once and kept open for the lifetime of the returned object.
        Only per-message index timestamps are loaded into memory; image bytes
        are read on demand when a specific index is accessed.

        Args:
            bag_path (Path | str): Path to the ``.bag`` file.
            img_topic (str): Topic name of the ``sensor_msgs/msg/Image`` stream.
        Returns:
            ImageDataOnDisk: Instance of this class.
        Raises:
            ValueError: If ``img_topic`` is not present in the bag.
        """
        typestore = get_typestore(Stores.ROS1_NOETIC)
        reader = Reader1(Path(bag_path))
        reader.open()

        conns = [c for c in reader.connections if c.topic == img_topic]
        if not conns:
            reader.close()
            raise ValueError(f"Topic {img_topic!r} not found in bag {bag_path}.")
        conn = conns[0]

        # Single pass: extract metadata from the first message and build a
        # {header_stamp_ns -> recording_time_ns} mapping for all messages.
        frame_id: str = ''
        height: int = 0
        width: int = 0
        encoding: ImageData.ImageEncoding = ImageData.ImageEncoding.RGB8
        pairs: List[tuple] = []  # (header_stamp: Decimal, recording_time_ns: int)

        num_msgs = conn.msgcount
        pbar = tqdm.tqdm(total=num_msgs, desc="Indexing Images...", unit=" msgs")
        for _, rec_ns, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            if not pairs:
                frame_id = msg.header.frame_id
                if conn.msgtype == ImageData._COMPRESSED_MSGTYPE:
                    first_image, encoding = ImageData._decode_compressed_image_msg(msg)
                    height, width = first_image.shape[:2]
                else:
                    height = msg.height
                    width = msg.width
                    encoding = ImageData.ImageEncoding.from_ros2_str(msg.encoding)
            stamp = msg.header.stamp
            h_stamp = Decimal(stamp.sec) + Decimal(stamp.nanosec) * Decimal('1e-9')
            pairs.append((h_stamp, rec_ns))
            pbar.update(1)
        pbar.close()

        pairs.sort(key=lambda p: p[0])
        timestamps: List[Decimal] = [p[0] for p in pairs]
        header_to_rec_ns: dict = {int(p[0] * Decimal('1e9')): p[1] for p in pairs}

        instance = cls(frame_id, timestamps, height, width, encoding, [], bag_reader=reader)
        instance.images = BagLazyImageArray(instance, reader, conn.id, timestamps,
                                            header_to_rec_ns, conn.msgtype, typestore)
        return instance

    # =========================================================================
    # ========================= Multi Data Methods ==========================
    # =========================================================================

    def _ensure_matching_image_shape(self, camera_data: CameraData) -> None:
        """
        Raise if ``camera_data``'s ``width``/``height`` don't match this
        ImageDataOnDisk's.

        Args:
            camera_data: The CameraData to check against this ImageDataOnDisk.

        Raises:
            ValueError: If the dimensions don't match.
        """

        if camera_data.width != self.width or camera_data.height != self.height:
            raise ValueError(
                f"camera_data dimensions ({camera_data.width}x{camera_data.height}) do not "
                f"match this ImageDataOnDisk's ({self.width}x{self.height}).")

    def crop_images_to_LiDAR_FOV(self, lidar_v_fov: Tuple[float, float],
                                  camera_data: CameraData) -> None:
        """
        Crop the top and bottom of every image so that the camera's vertical
        FOV matches the given LiDAR vertical FOV.

        The LiDAR vertical FOV is specified in degrees, measured from the
        horizontal plane: positive angles are above horizontal, negative
        angles are below. The crop is applied lazily via the transformation
        pipeline, so no images are read from disk until accessed.

        The crop row boundaries are computed from the camera intrinsic matrix
        ``K`` (specifically ``fy`` and ``cy``). For a pixel row ``v`` in the
        original image, the elevation angle is ``theta = -arctan((v - cy) / fy)``,
        so the row that maps to a given angle is ``v = cy - fy * tan(theta)``.

        No LiDAR-to-camera extrinsic transform is used, so this assumes the
        LiDAR and camera occupy the same position in 3D space (coincident
        centers) with zero relative roll and pitch, i.e. the LiDAR's
        horizontal plane (its 0° elevation reference) coincides with the
        camera's optical plane (the plane spanned by the camera's x-axis and
        forward/z-axis). In practice the two sensors are never truly
        colocated, but this crop is a good approximation as long as the
        physical offset between them is small relative to the distance to
        the scene (e.g. a few cm apart, observing a scene meters away). Any
        real mounting offset in roll, pitch, or position will make the crop
        systematically off, more so at short range. This function also only
        crops rows (vertical FOV); it does not perform any horizontal/azimuth
        cropping.

        Args:
            lidar_v_fov: ``(min_deg, max_deg)`` tuple giving the LiDAR's
                vertical angular range in degrees. Positive is above
                horizontal, negative is below. ``min_deg < max_deg`` is
                required.
            camera_data: A ``CameraData`` instance whose intrinsics are used
                to compute the crop boundaries. Its ``height``, ``K``, and
                ``P`` fields are updated to reflect the new image dimensions.

        Raises:
            ValueError: If ``min_deg >= max_deg``.
            ValueError: If the LiDAR FOV does not intersect the image's
                vertical extent, or if the computed crop has zero height.
        """

        if lidar_v_fov[0] >= lidar_v_fov[1]:
            raise ValueError(
                f"lidar_v_fov min ({lidar_v_fov[0]}) must be less than max ({lidar_v_fov[1]}).")

        self._ensure_matching_image_shape(camera_data)

        fy = float(camera_data.K[1, 1])
        cy = float(camera_data.K[1, 2])

        v_min_rad = np.radians(lidar_v_fov[0])
        v_max_rad = np.radians(lidar_v_fov[1])

        # v_max_deg is the highest elevation → smallest row number (closest to top)
        # v_min_deg is the lowest elevation → largest row number (closest to bottom)
        row_top    = int(np.floor(cy - fy * np.tan(v_max_rad)))
        row_bottom = int(np.ceil( cy - fy * np.tan(v_min_rad)))

        # Clamp to valid pixel range
        row_top    = max(0, row_top)
        row_bottom = min(self.height, row_bottom)

        new_height = row_bottom - row_top
        if new_height <= 0:
            raise ValueError(
                f"LiDAR FOV ({lidar_v_fov[0]}°, {lidar_v_fov[1]}°) does not intersect "
                f"the image's vertical extent (height={self.height}, cy={cy}, fy={fy}).")

        # Register the crop as a lazy transformation
        def _crop(image: np.ndarray) -> np.ndarray:
            return image[row_top:row_bottom, :]

        self.images.transformations.append(_crop)
        self.height = new_height

        # Update camera intrinsics to match the cropped image
        new_cy = cy - row_top
        camera_data.height = new_height
        camera_data.K[1, 2] = new_cy
        camera_data.P[1, 2] = new_cy

    @staticmethod
    def _build_undistort_map(K: np.ndarray, D: np.ndarray, R: np.ndarray, new_K: np.ndarray,
                              size_cv2: Tuple, distortion_model: CameraData.DistortionModel
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build ``cv2.remap`` maps that undistort (and, via ``R``, rectify)
        imagery for a single camera.

        Args:
            K: The camera's 3x3 intrinsic matrix.
            D: The camera's distortion coefficients.
            R: The 3x3 rectification rotation to apply (identity for mono).
            new_K: The target 3x3 camera matrix for the undistorted imagery.
            size_cv2: ``(width, height)`` of the imagery.
            distortion_model: The distortion model of the camera.

        Returns:
            Tuple of ``(map1, map2)`` as returned by
            ``cv2.initUndistortRectifyMap``/``cv2.fisheye.initUndistortRectifyMap``.

        Raises:
            NotImplementedError: If ``distortion_model`` is not supported.
        """

        if distortion_model == CameraData.DistortionModel.RADIAL_TANGENTIAL:
            return cv2.initUndistortRectifyMap(K, D, R, new_K, size_cv2, cv2.CV_32FC1)
        elif distortion_model == CameraData.DistortionModel.EQUIDISTANT:
            return cv2.fisheye.initUndistortRectifyMap(K, D.reshape(4, 1), R, new_K, size_cv2, cv2.CV_32FC1)
        else:
            raise NotImplementedError(
                f"Undistortion does not support distortion model {distortion_model}.")

    def undistort_imagery_mono(self, camera_data: CameraData) -> None:
        """
        Undistort every image using ``camera_data``'s already-computed target
        camera matrix (``P``'s intrinsic part), e.g. as produced by
        ``CameraData.from_user_mono``/``CameraData.from_kalibr_mono``.
        Afterwards, ``camera_data``'s ``K`` is updated to match and ``D`` is
        zeroed.

        The undistortion is applied lazily via the transformation pipeline,
        so no images are read from disk until accessed. Image dimensions are
        unchanged; only pixel content is remapped.

        Args:
            camera_data: The CameraData providing the distortion model,
                coefficients, and target ``P``. Its ``width``/``height`` must
                match this ImageDataOnDisk's. Its ``K`` and ``D`` are updated
                to describe the undistorted imagery.

        Raises:
            ValueError: If ``camera_data``'s dimensions don't match this
                ImageDataOnDisk's.
            NotImplementedError: If ``camera_data.distortion_model`` is not
                supported.
        """

        self._ensure_matching_image_shape(camera_data)
        size_cv2: Tuple = (self.width, self.height)
        new_K = camera_data.P[:3, :3].copy()

        map1, map2 = self._build_undistort_map(
            camera_data.K, camera_data.D, camera_data.R, new_K, size_cv2, camera_data.distortion_model)

        def _undistort(image: np.ndarray) -> np.ndarray:
            return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

        self.images.transformations.append(_undistort)

        camera_data.K = new_K
        camera_data.D = np.zeros_like(camera_data.D)

    @staticmethod
    def undistort_imagery_stereo(image_data_left: ImageDataOnDisk, image_data_right: ImageDataOnDisk,
                                 camera_data_left: CameraData, camera_data_right: CameraData) -> None:
        """
        Undistort and rectify a stereo image pair using ``camera_data_left``'s
        and ``camera_data_right``'s already-computed rectification (``R``) and
        rectified projection matrix (``P``), e.g. as produced by
        ``CameraData.from_kalibr_stereo``. Afterwards, each camera's ``K`` is
        updated to the rectified projection's intrinsics, ``D`` is zeroed,
        and ``R`` is reset to identity, since each image is now already
        sitting in its own rectified frame and no further rotation is needed
        to interpret it. ``P`` is left unchanged, since it still correctly
        describes the projection (including, for the right camera, the
        stereo baseline) of the now-rectified image.

        The undistortion/rectification is applied lazily via each
        ImageDataOnDisk's transformation pipeline, so no images are read from
        disk until accessed. Image dimensions are unchanged; only pixel
        content is remapped.

        Args:
            image_data_left: The left camera's imagery.
            image_data_right: The right camera's imagery.
            camera_data_left: The left camera's calibration, including its
                rectification ``R`` and rectified ``P``. Its ``width``/
                ``height`` must match ``image_data_left``'s. Its ``K``, ``D``,
                and ``R`` are updated to describe the undistorted imagery.
            camera_data_right: The right camera's calibration, matching
                ``image_data_right``, updated the same way as
                ``camera_data_left``.

        Raises:
            ValueError: If either CameraData's dimensions don't match its
                corresponding ImageDataOnDisk's, if
                ``camera_data_left.timeshift_cam_imu`` doesn't match
                ``camera_data_right.timeshift_cam_imu``, or if
                ``image_data_left`` and ``image_data_right`` don't have
                identical timestamps.
            NotImplementedError: If either camera's ``distortion_model`` is
                not supported.
        """

        if camera_data_left.timeshift_cam_imu != camera_data_right.timeshift_cam_imu:
            raise ValueError(
                "camera_data_left.timeshift_cam_imu "
                f"({camera_data_left.timeshift_cam_imu}) does not match "
                f"camera_data_right.timeshift_cam_imu ({camera_data_right.timeshift_cam_imu}). "
                "Call CameraData.align_ImageData_and_CameraData_to_imu_ts() on both cameras first.")

        if not np.array_equal(image_data_left.timestamps, image_data_right.timestamps):
            raise ValueError(
                "image_data_left and image_data_right must have identical timestamps.")

        for image_data, camera_data in ((image_data_left, camera_data_left),
                                        (image_data_right, camera_data_right)):
            image_data._ensure_matching_image_shape(camera_data)
            size_cv2: Tuple = (image_data.width, image_data.height)
            new_K = camera_data.P[:3, :3].copy()

            map1, map2 = ImageDataOnDisk._build_undistort_map(
                camera_data.K, camera_data.D, camera_data.R, new_K, size_cv2, camera_data.distortion_model)

            def _undistort(image: np.ndarray, map1=map1, map2=map2) -> np.ndarray:
                return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

            image_data.images.transformations.append(_undistort)

            camera_data.K = new_K
            camera_data.R = np.eye(3, dtype=np.float64)
            camera_data.D = np.zeros_like(camera_data.D)
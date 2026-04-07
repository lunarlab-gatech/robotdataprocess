from __future__ import annotations
from ...conversion_utils import col_to_dec_arr
from .ImageData import ImageData
import cv2
import copy
import numpy as np
from pathlib import Path
from PIL import Image
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

@typechecked
class ImageDataOnDisk(ImageData):
    """
    Image data loaded lazily from disk, reading each image only when accessed.
    Uses a ``LazyImageArray`` that loads PNG or ``.npy`` files on demand,
    keeping memory usage low for large image sequences. Supports loading from
    ``.png`` and ``.npy`` folders where filenames encode timestamps.
    """
        
    images: LazyImageArray # Not initalized here, but put here for visual code highlighting

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], height: int, width: int, 
                 encoding: ImageData.ImageEncoding, image_paths: List[Path], transformations: Union[List[Callable], None] = None):
        
        super().__init__(frame_id, timestamps, height, width, encoding, None)
        transformations_copy = copy.deepcopy(transformations) if transformations else []
        self.images = LazyImageArray(self, image_paths, transformations_copy)

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

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def crop_images_to_LiDAR_FOV(self, lidar_v_fov: Tuple[float, float],
                                  camera_data: Any) -> None:
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

        # Import here to avoid circular imports
        from ..CameraData import CameraData
        if not isinstance(camera_data, CameraData):
            raise TypeError(f"camera_data must be a CameraData instance, got {type(camera_data)}.")

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
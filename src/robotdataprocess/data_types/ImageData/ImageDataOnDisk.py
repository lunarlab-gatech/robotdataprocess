from __future__ import annotations
from ...conversion_utils import col_to_dec_arr
from .ImageData import ImageData
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typeguard import typechecked
from typing import Union, List, Callable

class LazyImageArray:
    """A read-only array-like interface that loads PNG images from disk on demand."""
    
    container: 'ImageDataOnDisk'
    image_paths: List[Path]
    transformations: List[Callable]

    def __init__(self, container: 'ImageDataOnDisk', image_paths: List[Path], transformations: List[Callable] = []):
        self.container = container
        self.image_paths = image_paths
        self.transformations = transformations

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
                 encoding: ImageData.ImageEncoding, image_paths: List[Path], transformations: List[Callable] = []):
        super().__init__(frame_id, timestamps, height, width, encoding, None)
        self.images = LazyImageArray(self, image_paths, transformations)

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
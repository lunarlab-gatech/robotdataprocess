from __future__ import annotations

from ...conversion_utils import col_to_dec_arr
from .ImageData import ImageData
import numpy as np
from pathlib import Path
from PIL import Image
from typeguard import typechecked
from typing import Union, List

@typechecked
class ImageDataOnDisk(ImageData):

    class LazyImageArray:
        """A read-only array-like interface that loads PNG images from disk on demand."""
        
        height: int
        width: int
        encoding: ImageData.ImageEncoding
        image_paths: List[Path]

        def __init__(self, height: int, width: int, encoding: ImageData.ImageEncoding, image_paths: List[Path]):
            self.height = height
            self.width = width
            self.encoding = encoding
            self.image_paths = image_paths

        def __getitem__(self, idx) -> np.ndarray:
            # Handle boolean masking (used by your crop_data function)
            if isinstance(idx, np.ndarray) and idx.dtype == bool:
                new_paths = [p for p, keep in zip(self.image_paths, idx) if keep]
                return self.__class__(self.height, self.width, self.encoding, new_paths)

            # Handle slicing (e.g., images[0:10])
            if isinstance(idx, slice):
                return self.__class__(self.height, self.width, self.encoding, self.image_paths[idx])

            # Handle single integer indexing (loading the actual image)
            path: Path = self.image_paths[idx]
            return np.array(Image.open(str(path)), dtype=self.dtype)

        def __setitem__(self, idx, value):
            raise RuntimeError("This LazyPNGArray is read-only; writes are forbidden.")
    
        def __len__(self):
            return len(self.image_paths)

        @property
        def shape(self):
            _, channels = ImageData.ImageEncoding.to_dtype_and_channels(self.encoding)
            if channels == 1:
                return (len(self), self.height, self.width)
            else:
                return (len(self), self.height, self.width, 3)

        @property
        def dtype(self):
            dtype_val, _ = ImageData.ImageEncoding.to_dtype_and_channels(self.encoding)
            return dtype_val
        
    images: ImageDataOnDisk.LazyImageArray # Not initalized here, but put here for visual code highlighting

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list], images: ImageDataOnDisk.LazyImageArray):
        super().__init__(frame_id, timestamps, images.height, images.width, images.encoding, images)

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
        
        # Return an ImageDataOnDisk class
        return cls(frame_id, timestamps_sorted, cls.LazyImageArray(first_image.height, first_image.width, encoding, all_image_files_sorted))
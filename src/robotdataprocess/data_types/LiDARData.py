from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from .Data import Data
from decimal import Decimal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typeguard import typechecked
from typing import Union, List
import tqdm

@typechecked
class LiDARData(Data):
    """
    LiDAR Data class that contains LiDAR-specific attributes and methods.
    Inherits from the generic Data class.
    """

    point_clouds: NDArray[Decimal] # (T, N, 3) array of point clouds, with assumed (x, y, z) ordering

    def __init__(self, frame_id: str, timestamps: np.ndarray | list, point_clouds: NDArray) -> None:
        super().__init__(frame_id, timestamps)
        self.point_clouds = point_clouds

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    def from_npy_files(cls, npy_folder_path: Union[Path, str], frame_id: str) -> LiDARData:
        """
        Load LiDAR data from a series of .npy files in a specified folder,
        where the file names correspond to timestamps.

        Args:
            npy_folder_path: Path to the folder.
            frame_id: The frame ID for the LiDAR data.
        Returns:
            LiDARData: An instance of LiDARData populated with the loaded data.
        """

        # Get all npy files in the designated folder (sorted)
        all_npy_files: List[str] = [str(p) for p in Path(npy_folder_path).glob("*.npy")]
        print(f"Found {len(all_npy_files)} .npy files in folder {npy_folder_path}")

        # Extract the timestamps and sort them
        timestamps = col_to_dec_arr([s.split('/')[-1][:-4] for s in all_npy_files])
        sorted_indices = np.argsort(timestamps)
        timestamps_sorted = timestamps[sorted_indices]

        # Use sorted_indices to sort all_image_files in the same way
        all_npy_files_sorted = [all_npy_files[i] for i in sorted_indices]

        # Check the point cloud shape from the first file
        first_pc = np.load(all_npy_files_sorted[0], 'r')
        assert len(first_pc.shape) == 2
        assert first_pc.shape[1] == 3

        # Load all the point clouds into a single array
        point_clouds = np.zeros((len(all_npy_files_sorted), first_pc.shape[0], 3), dtype=np.float64)
        pbar = tqdm.tqdm(total=len(all_npy_files_sorted), desc="Extracting Point Clouds...", unit=" files")
        for i, path in enumerate(all_npy_files_sorted):
            point_clouds[i] = np.load(path, 'r')
            assert point_clouds[i].shape[0] == first_pc.shape[0]
            pbar.update()

        # Return an LiDARData class
        return cls(frame_id, timestamps_sorted, point_clouds)
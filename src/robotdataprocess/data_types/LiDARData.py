from __future__ import annotations

from ..conversion_utils import col_to_dec_arr
from .Data import Data, CoordinateFrame
from decimal import Decimal
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typeguard import typechecked
from typing import Union, List, Tuple
import tqdm

@typechecked
class LiDARData(Data):
    """
    LiDAR Data class that contains LiDAR-specific attributes and methods.
    Inherits from the generic Data class.
    """

    point_clouds: NDArray[Decimal] # (T, N, 3) array of point clouds, with assumed (x, y, z) ordering
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: np.ndarray | list, point_clouds: NDArray, frame: CoordinateFrame) -> None:
        super().__init__(frame_id, timestamps)
        self.point_clouds = point_clouds
        self.frame = frame

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    def from_npy_files(cls, npy_folder_path: Union[Path, str], frame_id: str, frame: CoordinateFrame) -> LiDARData:
        """
        Load LiDAR data from a series of .npy files in a specified folder,
        where the file names correspond to timestamps.

        Args:
            npy_folder_path: Path to the folder.
            frame_id: The frame ID for the LiDAR data.
            frame: The Coordinate frame of the LiDAR data.
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

        print("SHAPE OF point clouds: ", point_clouds.shape)

        # Return an LiDARData class
        return cls(frame_id, timestamps_sorted, point_clouds, frame)

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
            R_NED = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]])
            for i in range(self.point_clouds.shape[0]):
                self.point_clouds[i] = (R_NED @ self.point_clouds[i].T).T
            self.frame = CoordinateFrame.FLU

        else:
            raise RuntimeError(f"LiDARData class is in an unexpected frame: {self.frame}!")

    # =========================================================================
    # ============================ Visualization ============================== 
    # =========================================================================  

    @typechecked
    def visualize(self, interval_ms: int = 1000, xlim: Tuple[float, float] = (-50.0, 50.0), ylim: Tuple[float, float] = (-50.0, 50.0), 
                  zlim: Tuple[float, float] = (-50.0, 50.0)):
        """
        Visualizes the raw LiDAR data over time.
        """

        T, N, _ = self.point_clouds.shape

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter([], [], [], s=2)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        def update(frame: int):
            pts = np.asarray(self.point_clouds[frame], dtype=float)
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            ax.set_title(f"LiDAR Frame {frame}/{T-1}")
            return scatter,

        ani = FuncAnimation(
            fig,
            update,
            frames=T,
            interval=interval_ms,
            blit=False,
            repeat=True,
        )

        plt.show()
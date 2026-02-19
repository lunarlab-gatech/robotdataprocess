from __future__ import annotations

from .Data import Data, CoordinateFrame, TransformType
import json
import matplotlib.pyplot as plt
import numpy as np
from typeguard import typechecked
from typing import List, Tuple, Union
from scipy.spatial.transform import Rotation as R
import yaml

@typechecked
class TransformationData(Data):
   
    child_frame_id: str
    translation: np.ndarray  # (3) translation vector
    orientation: np.ndarray  # (4) quaternion in xyzw format
    frame: CoordinateFrame

    def __init__(self, frame_id: str, child_frame_id: str, translation: np.ndarray, orientation: np.ndarray, frame: CoordinateFrame):
                 
        super().__init__(frame_id=frame_id)
        self.child_frame_id = child_frame_id
        self.translation = translation
        self.orientation = orientation
        self.frame = frame

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_HERCULES_settings_json(cls, json_path: str, robot_name: str, sensor_type: str, sensor_name: str) -> TransformationData:
        """
        Load a transformation from a HERCULES settings JSON.
        """

        # Open the json
        with open(json_path, "r") as f:
            settings = json.load(f)

        # Extract robot config
        vehicles = settings.get("Vehicles", {})
        if robot_name not in vehicles:
            raise KeyError(f"Robot '{robot_name}' not found in Vehicles")
        robot = vehicles[robot_name]

        # Extract sensor type block
        if sensor_type.lower() == "camera":
            block = robot.get("Cameras", {})
        elif sensor_type.lower() == "sensor":
            block = robot.get("Sensors", {})
        else:
            raise ValueError("sensor_type must be 'Camera' or 'Sensor'")

        # Extract sensor config
        if sensor_name not in block:
            raise KeyError(
                f"{sensor_type} '{sensor_name}' not found on robot '{robot_name}'"
            )
        data = block[sensor_name]

        # Extract transformation
        translation = np.array([data["X"], data["Y"], data["Z"]], dtype=float)
        rotation = R.from_euler(seq="xyz", angles=[data["Roll"], data["Pitch"], data["Yaw"]], degrees=True,)
        orientation = rotation.as_quat()
        
        # Create the class
        return cls(
            frame_id=robot_name,
            child_frame_id=sensor_name,
            translation=translation,
            orientation=orientation,
            frame=CoordinateFrame.NED,
        )

    @classmethod
    def from_GrAco_yaml(cls, yaml_path: str, transform_name: str) -> TransformationData:
        """
        Load a transformation from a GrAco calibration YAML file.

        Args:
            yaml_path: Path to the GrAco YAML file.
            transform_name: Name of the transform key (e.g. 'T_Imu_Lidar').
        """
    
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if transform_name not in data:
            raise KeyError(f"Transform '{transform_name}' not found in {yaml_path}")

        transform = data[transform_name]
        rows = transform["rows"]
        cols = transform["cols"]
        if rows != 4 or cols != 4:
            raise ValueError(f"Expected 4x4 transformation matrix, got {rows}x{cols}")
        flat_data = transform["data"]
        matrix = np.array(flat_data, dtype=float).reshape(rows, cols)

        # Extract frame_id and child_frame_id from transform name (e.g. T_Imu_Lidar -> Imu, Lidar)
        parts = transform_name.split("_")
        frame_id = parts[1]
        child_frame_id = "_".join(parts[2:])

        return cls.from_matrix(frame_id, child_frame_id, matrix, CoordinateFrame.ENU)

    @classmethod
    def from_matrix(cls, frame_id: str, child_frame_id: str, matrix: np.ndarray, frame: CoordinateFrame) -> TransformationData:
        """
        Creates a TransformationData object from a 4x4 transformation matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")
        
        translation = matrix[0:3, 3]
        rotation_matrix = matrix[0:3, 0:3]
        orientation = R.from_matrix(rotation_matrix).as_quat()

        return cls(frame_id, child_frame_id, translation, orientation, frame)
    
    @classmethod
    def optical_wrt_camera(cls, frame: CoordinateFrame, frame_id: str = "camera", child_frame_id: str = "optical") -> TransformationData:
        """ Get H_C_to_W in a specified frame """
        if frame == CoordinateFrame.NED:
            rot = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
        elif frame == CoordinateFrame.FLU:
            rot = np.array([[ 0,  0, 1],
                            [-1,  0, 0],
                            [ 0, -1, 0]])
        elif frame == CoordinateFrame.ENU:
            rot = np.array([[1,  0, 0],
                            [0,  0, 1],
                            [0, -1, 0]])
        else:
            raise RuntimeError(f"optical_wrt_camera not yet implemented for CoordinateFrame {frame}.")

        orientation = R.from_matrix(rot).as_quat()
        return cls(frame_id, child_frame_id, np.zeros(3), orientation, frame)

    # =========================================================================
    # ========================== Transform Methods ============================
    # =========================================================================

    def to_coordinate_frame(self, target_frame: CoordinateFrame, transform_type: TransformType = TransformType.ROTATION) -> TransformationData:
        """
        Returns a new TransformationData in the target coordinate frame.
        Currently only supports NED to FLU.

        Args:
            target_frame: The desired coordinate frame.
            transform_type: How to apply the frame change.
                ROTATION: Left-multiplies the frame change rotation onto the
                    translation and orientation (default, original behaviour).
                CHANGE_OF_BASIS: Applies a similarity transform T_new = R T R^{-1},
                    re-expressing the same physical transformation in the new basis.
        """
        if self.frame == target_frame:
            return TransformationData(self.frame_id, self.child_frame_id, self.translation.copy(), self.orientation.copy(), self.frame)

        if self.frame == CoordinateFrame.NED and target_frame == CoordinateFrame.FLU:
            # The frame change rotation: 180 degrees around X
            R_frame = R.from_euler('x', 180, degrees=True)

            if transform_type == TransformType.ROTATION:
                # Apply as a rotation: R_frame * t, R_frame * q
                new_translation = self.translation.copy()
                new_translation[1] *= -1  # Y becomes -Y
                new_translation[2] *= -1  # Z becomes -Z
                new_orientation = (R_frame * R.from_quat(self.orientation)).as_quat()

            elif transform_type == TransformType.CHANGE_OF_BASIS:
                # Similarity transform: T_new = R * T * R^{-1}
                R_mat = R_frame.as_matrix()
                T = self.as_matrix()
                R_4x4 = np.identity(4)
                R_4x4[0:3, 0:3] = R_mat
                R_inv_4x4 = np.identity(4)
                R_inv_4x4[0:3, 0:3] = R_mat.T
                T_new = R_4x4 @ T @ R_inv_4x4
                new_translation = T_new[0:3, 3]
                new_orientation = R.from_matrix(T_new[0:3, 0:3]).as_quat()

            return TransformationData(self.frame_id, self.child_frame_id, new_translation, new_orientation, CoordinateFrame.FLU)
        else:
            raise NotImplementedError(f"Transformation from {self.frame} to {target_frame} is not implemented.")

    def invert(self) -> TransformationData:
        """
        Returns the inverse transformation such that self @ self.invert() == Identity.
        Swaps frame_id and child_frame_id.
        """
        rot = R.from_quat(self.orientation).as_matrix()
        inv_rot = rot.T
        inv_translation = -inv_rot @ self.translation
        inv_orientation = R.from_matrix(inv_rot).as_quat()
        return TransformationData(self.child_frame_id, self.frame_id, inv_translation, inv_orientation, self.frame)

    def apply_transformation_right_side(self, other: TransformationData) -> TransformationData:
        """
        Applies another transformation to the right side of this transformation.
        Effectively, this transformation becomes self @ other.
        """
        if self.frame != other.frame:
            raise ValueError(f"Coordinate frames must match for right-side transformation: {self.frame} vs {other.frame}")
        
        if self.child_frame_id != other.frame_id:
            raise ValueError(f"Child frame ID of self must match frame ID of other for right-side transformation: {self.child_frame_id} vs {other.frame_id}")

        # Convert to 4x4 matrices
        self_matrix = self.as_matrix()
        other_matrix = other.as_matrix()

        # Multiply the matrices
        new_matrix = self_matrix @ other_matrix

        # Return the result
        new_rotation_matrix = new_matrix[0:3, 0:3]
        return TransformationData(self.frame_id, other.child_frame_id, new_matrix[0:3, 3], R.from_matrix(new_rotation_matrix).as_quat(), self.frame)

    # =========================================================================
    # =========================== Export Methods ==============================
    # =========================================================================

    def as_matrix(self) -> np.ndarray:
        matrix = np.identity(4)
        matrix[0:3, 0:3] = R.from_quat(self.orientation).as_matrix()
        matrix[0:3, 3] = self.translation
        return matrix
    
    # =========================================================================
    # ======================== Visualization Methods===========================
    # =========================================================================

    @staticmethod
    def visualize(transformations: List[TransformationData], axes_length: float = 1.0):
        """ Visualize multiple transformations in the same 3D space """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract data and plot
        points = []
        transformations.append(TransformationData.from_matrix("World", "World", np.eye(4), CoordinateFrame.FLU))
        for trans in transformations:
            pos = trans.translation
            rot = R.from_quat(trans.orientation)

            # Define unit vectors for X, Y, Z in local frame
            x_axis = rot.apply([1, 0, 0])
            y_axis = rot.apply([0, 1, 0])
            z_axis = rot.apply([0, 0, 1])

            # Plot axes
            ax.quiver(*pos, *x_axis, length=axes_length, color='r', normalize=True, linewidth=0.8)
            ax.quiver(*pos, *y_axis, length=axes_length, color='g', normalize=True, linewidth=0.8)
            ax.quiver(*pos, *z_axis, length=axes_length, color='b', normalize=True, linewidth=0.8)

            # Collect endpoints for bounds
            points.append(pos)
            points.append(pos + x_axis)
            points.append(pos + y_axis)
            points.append(pos + z_axis)

        # Compute bounds
        points = np.vstack(points)
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        max_range = (max_xyz - min_xyz).max() / 2.0

        # Set equal axis limits
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        # Set labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Equal aspect ratio (Matplotlib ≥ 3.3)
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass  # older matplotlib

        # Show the plot
        plt.tight_layout()
        plt.show()
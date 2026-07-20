from __future__ import annotations

from .Data import CoordinateFrame, ROSMsgLibType, TransformType
from .SequentialData import SequentialData
import decimal
from decimal import Decimal
import json
from ..ModuleImporter import ModuleImporter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rosbags.typesys import Stores, get_typestore
from typeguard import typechecked
from typing import Any, List, Tuple, Union
from scipy.spatial.transform import Rotation as R
import yaml

@typechecked
class TransformationData(SequentialData):

    child_frame_id: str
    translation: np.ndarray  # (3) translation vector
    orientation: np.ndarray  # (4) quaternion in xyzw format
    frame: CoordinateFrame

    def __init__(self, frame_id: str, child_frame_id: str, translation: np.ndarray, orientation: np.ndarray, frame: CoordinateFrame):

        super().__init__(frame_id=frame_id, timestamps=[Decimal('0')])
        self.child_frame_id = child_frame_id
        self.translation = translation
        self.orientation = orientation
        self.frame = frame

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in TransformationData. """
        pass

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_HERCULES_settings_json(cls, json_path: str, robot_name: str, sensor_type: str, sensor_name: str) -> TransformationData:
        """
        Load a transformation from a HERCULES settings JSON.

        Args:
            json_path: Path to the HERCULES settings JSON file.
            robot_name: Name of the vehicle in the JSON (under ``Vehicles``).
            sensor_type: Either ``"Camera"`` or ``"Sensor"``.
            sensor_name: Name of the sensor within the type block.

        Returns:
            TransformationData: Instance with NED coordinate frame.

        Raises:
            KeyError: If the robot or sensor is not found in the JSON.
            ValueError: If ``sensor_type`` is not ``"Camera"`` or ``"Sensor"``.
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
    def from_kalibr(cls, yaml_path: Union[Path, str], cam_name: str, transform_name: str,
                     frame: CoordinateFrame) -> TransformationData:
        """
        Load a transformation from a kalibr camera calibration YAML file.

        Kalibr stores transforms under a camera block (e.g. ``cam0``) as a
        4x4 nested list following the convention ``p_A = T_A_B @ p_B``, where
        ``A`` and ``B`` are given by the transform name (e.g. ``T_cam_imu``
        maps IMU-frame points into the camera frame). The token ``cam`` (or
        ``cn``) in the transform name is replaced with ``cam_name`` to
        recover the frame_id.

        Args:
            yaml_path: Path to the kalibr YAML file.
            cam_name: Name of the camera block (e.g. ``'cam0'``).
            transform_name: Name of the transform key within the camera
                block (e.g. ``'T_cam_imu'``).
            frame: The coordinate frame of this transformation.

        Raises:
            KeyError: If ``cam_name`` or ``transform_name`` is not found in
                the YAML.
            ValueError: If the transform is not a 4x4 matrix.
        """

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if cam_name not in data:
            raise KeyError(f"Camera '{cam_name}' not found in {yaml_path}")
        cam = data[cam_name]

        if transform_name not in cam:
            raise KeyError(f"Transform '{transform_name}' not found for camera '{cam_name}' in {yaml_path}")

        matrix = np.array(cam[transform_name], dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transformation matrix, got {matrix.shape}")

        # Extract frame_id and child_frame_id from transform name (e.g. T_cam_imu -> cam0, imu)
        parts = transform_name.split("_")[1:]
        frame_id = cam_name if parts[0] in ("cam", "cn") else parts[0]
        child_frame_id = "_".join(parts[1:])

        return cls.from_matrix(frame_id, child_frame_id, matrix, frame)

    @classmethod
    def from_GrAco_yaml(cls, yaml_path: Union[Path, str], transform_name: str) -> TransformationData:
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

        Args:
            frame_id: The parent frame ID.
            child_frame_id: The child frame ID.
            matrix: A 4x4 homogeneous transformation matrix.
            frame: The coordinate frame of this transformation.

        Returns:
            TransformationData: Instance of this class.

        Raises:
            ValueError: If ``matrix`` is not 4x4.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")

        translation = matrix[0:3, 3]
        rotation_matrix = matrix[0:3, 0:3]
        orientation = R.from_matrix(rotation_matrix).as_quat()

        return cls(frame_id, child_frame_id, translation, orientation, frame)

    @classmethod
    def optical_wrt_camera(cls, frame: CoordinateFrame, frame_id: str = "camera", child_frame_id: str = "optical") -> TransformationData:
        """
        Get the optical-frame-w.r.t.-camera transformation in a specified coordinate frame.

        Args:
            frame: The coordinate frame convention (NED, FLU, or ENU).
            frame_id: The parent frame ID.
            child_frame_id: The child frame ID.

        Returns:
            TransformationData: Pure rotation (zero translation) from camera to optical.

        Raises:
            RuntimeError: If ``frame`` is not supported.
        """
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

        Returns:
            TransformationData: The inverse transformation.
        """
        rot = R.from_quat(self.orientation).as_matrix()
        inv_rot = rot.T
        inv_translation = -inv_rot @ self.translation
        inv_orientation = R.from_matrix(inv_rot).as_quat()
        return TransformationData(self.child_frame_id, self.frame_id, inv_translation, inv_orientation, self.frame)

    def apply_transformation_right_side(self, other: TransformationData) -> TransformationData:
        """
        Applies another transformation to the right side of this transformation.
        Effectively computes ``self @ other``.

        Args:
            other: The transformation to compose on the right.

        Returns:
            TransformationData: The composed transformation with ``self.frame_id``
            and ``other.child_frame_id``.

        Raises:
            ValueError: If coordinate frames do not match or if
                ``self.child_frame_id != other.frame_id``.
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
        """
        Returns the 4x4 homogeneous transformation matrix.

        Returns:
            np.ndarray: A 4x4 matrix with rotation and translation.
        """
        matrix = np.identity(4)
        matrix[0:3, 0:3] = R.from_quat(self.orientation).as_matrix()
        matrix[0:3, 3] = self.translation
        return matrix

    # =========================================================================
    # ======================== Visualization Methods===========================
    # =========================================================================

    @staticmethod
    def visualize(transformations: List[TransformationData], axes_length: float = 1.0):
        """
        Visualize multiple transformations in the same 3D space.

        Args:
            transformations: List of TransformationData objects to plot.
                An identity world frame is appended automatically.
            axes_length: Length of the plotted orientation axes in meters.
        """
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

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """
        Return the ROS message type class for a TFMessage message.

        Args:
            lib_type: Which ROS message library to use.

        Returns:
            The ROS message type class for ``tf2_msgs/TFMessage``.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            return typestore.types['tf2_msgs/msg/TFMessage'].__msgtype__
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            return ModuleImporter.get_module_attribute('tf2_msgs.msg', 'TFMessage')
        else:
            raise NotImplementedError(
                f"Unsupported ROSMsgLibType {lib_type} for TransformationData.get_ros_msg_type()!")

    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int = 0):
        """
        Build a TFMessage ROS message from this transformation data using the
        timestamp stored at index ``i``.

        Args:
            lib_type: Which ROS message library to use.
            i: Index into the timestamps array (default 0).

        Returns:
            A populated ``tf2_msgs/TFMessage`` containing one ``TransformStamped``.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

        seconds = int(self.timestamps[i])
        nanoseconds = int((self.timestamps[i] - self.timestamps[i].to_integral_value(
            rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN))

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            TFMessage = typestore.types['tf2_msgs/msg/TFMessage']
            TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
            Transform = typestore.types['geometry_msgs/msg/Transform']
            Header = typestore.types['std_msgs/msg/Header']
            Time = typestore.types['builtin_interfaces/msg/Time']
            Vector3 = typestore.types['geometry_msgs/msg/Vector3']
            Quaternion = typestore.types['geometry_msgs/msg/Quaternion']

            return TFMessage(transforms=[
                TransformStamped(
                    header=Header(
                        stamp=Time(sec=seconds, nanosec=nanoseconds),
                        frame_id=self.frame_id),
                    child_frame_id=self.child_frame_id,
                    transform=Transform(
                        translation=Vector3(
                            x=float(self.translation[0]),
                            y=float(self.translation[1]),
                            z=float(self.translation[2])),
                        rotation=Quaternion(
                            x=float(self.orientation[0]),
                            y=float(self.orientation[1]),
                            z=float(self.orientation[2]),
                            w=float(self.orientation[3]))))])

        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            TFMessage = ModuleImporter.get_module_attribute('tf2_msgs.msg', 'TFMessage')
            TransformStamped = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'TransformStamped')
            Transform = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Transform')
            Vector3 = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Vector3')
            Quaternion = ModuleImporter.get_module_attribute('geometry_msgs.msg', 'Quaternion')
            Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')

            ts = TransformStamped()
            ts.header = Header()
            if lib_type == ROSMsgLibType.RCLPY:
                Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                ts.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
            else:
                rospy = ModuleImporter.get_module('rospy')
                ts.header.stamp = rospy.Time(secs=seconds, nsecs=nanoseconds)
            ts.header.frame_id = self.frame_id
            ts.child_frame_id = self.child_frame_id
            ts.transform = Transform()
            ts.transform.translation = Vector3()
            ts.transform.translation.x = float(self.translation[0])
            ts.transform.translation.y = float(self.translation[1])
            ts.transform.translation.z = float(self.translation[2])
            ts.transform.rotation = Quaternion()
            ts.transform.rotation.x = float(self.orientation[0])
            ts.transform.rotation.y = float(self.orientation[1])
            ts.transform.rotation.z = float(self.orientation[2])
            ts.transform.rotation.w = float(self.orientation[3])

            msg = TFMessage()
            msg.transforms = [ts]
            return msg

        else:
            raise NotImplementedError(
                f"Unsupported ROSMsgLibType {lib_type} for TransformationData.get_ros_msg()!")

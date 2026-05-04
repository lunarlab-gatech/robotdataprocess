from __future__ import annotations

from .Data import ROSMsgLibType
from .SequentialData import SequentialData
import decimal
from decimal import Decimal
from enum import Enum
from ..ModuleImporter import ModuleImporter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from rosbags.rosbag1 import Reader as Reader1
from rosbags.typesys import Stores, get_typestore
from typeguard import typechecked
from typing import Any, Optional, Tuple, Union
import yaml
from .ImageData.ImageData import ImageData


@typechecked
class CameraData(SequentialData):
    """
    Camera calibration data that can be published as ``sensor_msgs/CameraInfo``
    ROS messages.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        distortion_model: The distortion model used for this camera.
        K: 3x3 camera intrinsic matrix (row-major).
        D: Distortion coefficients array (length depends on model).
        R: 3x3 rectification matrix (row-major).
        P: 3x4 projection matrix (row-major).
    """

    class DistortionModel(Enum):
        """
        Enum for supported camera distortion models.

        Attributes:
            RADIAL_TANGENTIAL: Radial and tangential distortion model,
                corresponding to ``"plumb_bob"`` in ROS.
        """

        RADIAL_TANGENTIAL = 0

        @staticmethod
        def to_ros_str(model: CameraData.DistortionModel) -> str:
            """
            Convert a DistortionModel enum value to its ROS string representation.

            Args:
                model: The distortion model to convert.

            Returns:
                The ROS distortion model string.

            Raises:
                NotImplementedError: If ``model`` has no ROS string mapping.
            """

            if model == CameraData.DistortionModel.RADIAL_TANGENTIAL:
                return "plumb_bob"
            else:
                raise NotImplementedError(
                    f"CameraData.DistortionModel.{model} has no ROS string mapping!")

        @classmethod
        def from_ros_str(cls, model_str: str) -> CameraData.DistortionModel:
            """
            Convert a ROS distortion model string to a DistortionModel enum value.

            Args:
                model_str: The ROS distortion model string (e.g. ``"plumb_bob"``).

            Returns:
                The corresponding DistortionModel enum value.

            Raises:
                NotImplementedError: If ``model_str`` is not a recognised ROS
                    distortion model string.
            """

            if model_str == "plumb_bob":
                return cls.RADIAL_TANGENTIAL
            else:
                raise NotImplementedError(
                    f"ROS distortion model string '{model_str}' is not supported!")

        @classmethod
        def from_kalibr_str(cls, model_str: str) -> CameraData.DistortionModel:
            """
            Convert a kalibr distortion model string to a DistortionModel enum value.

            Args:
                model_str: The kalibr distortion model string
                    (e.g. ``"radial-tangential"``).

            Returns:
                The corresponding DistortionModel enum value.

            Raises:
                NotImplementedError: If ``model_str`` is not a recognised kalibr
                    distortion model string.
            """

            if model_str.lower().replace('_', '-') == 'radial-tangential':
                return cls.RADIAL_TANGENTIAL
            else:
                raise NotImplementedError(
                    f"kalibr distortion model string '{model_str}' is not supported!")

    width: int
    height: int
    distortion_model: CameraData.DistortionModel
    K: NDArray  # shape (3, 3)
    D: NDArray  # shape (N,)
    R: NDArray  # shape (3, 3)
    P: NDArray  # shape (3, 4)

    def __init__(self, frame_id: str, width: int, height: int,
                 distortion_model: CameraData.DistortionModel,
                 K: Union[NDArray, list],
                 D: Union[NDArray, list],
                 R: Union[NDArray, list],
                 P: Union[NDArray, list]):

        super().__init__(frame_id, [Decimal('0')])
        self.width = width
        self.height = height
        self.distortion_model = distortion_model
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.D = np.array(D, dtype=np.float64).flatten()
        self.R = np.array(R, dtype=np.float64).reshape(3, 3)
        self.P = np.array(P, dtype=np.float64).reshape(3, 4)

        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in CameraData. """
        pass

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def sync_to_ImageData(self, image_data: ImageData) -> None:
        """
        Set this object's timestamps to exactly match those of ``image_data``.

        This ensures that CameraInfo messages are published at the same times
        as their corresponding Image messages.

        Args:
            image_data: The ImageData whose timestamps to copy.
        """
        self.timestamps = np.array(image_data.timestamps)

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_user_mono(cls, frame_id: str, width: int, height: int,
                       fx: float, fy: float, cx: float, cy: float,
                       distortion_model: Union[CameraData.DistortionModel, None] = None,
                       D: Union[NDArray, list, None] = None) -> CameraData:
        """
        Create a CameraData instance for a monocular camera from individual
        calibration parameters.

        R is fixed to the identity matrix (no stereo rectification). P is
        derived from K by appending a zero fourth column.

        Args:
            frame_id: The camera's optical frame ID.
            width: Image width in pixels.
            height: Image height in pixels.
            fx: Focal length in x (pixels).
            fy: Focal length in y (pixels).
            cx: Principal point x coordinate (pixels).
            cy: Principal point y coordinate (pixels).
            distortion_model: The distortion model. Defaults to
                ``DistortionModel.RADIAL_TANGENTIAL``.
            D: Distortion coefficients. Defaults to all-zeros (5 coefficients
                for ``RADIAL_TANGENTIAL``).

        Returns:
            CameraData: Instance populated with the provided calibration.
        """

        if distortion_model is None:
            distortion_model = CameraData.DistortionModel.RADIAL_TANGENTIAL

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        if D is None:
            D = np.zeros(5, dtype=np.float64)

        R = np.eye(3, dtype=np.float64)

        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = K

        return cls(frame_id=frame_id, width=width, height=height,
                   distortion_model=distortion_model,
                   K=K, D=D, R=R, P=P)

    @classmethod
    def from_kalibr_mono(cls, yaml_path: Union[str, Path], cam_name: str) -> CameraData:
        """
        Load a monocular camera calibration from a kalibr YAML file.

        The YAML entry for ``cam_name`` must contain:

        - ``intrinsics``: ``[fu, fv, cu, cv]``
        - ``distortion_coeffs``: distortion coefficient list
        - ``distortion_model``: e.g. ``"radial-tangential"``
        - ``resolution``: ``[width, height]``

        R is fixed to the identity matrix and P is derived from K by
        appending a zero fourth column.

        Args:
            yaml_path: Path to the kalibr YAML calibration file.
            cam_name: Camera key within the YAML (e.g. ``"cam0"``).

        Returns:
            CameraData: Instance populated with the loaded calibration.

        Raises:
            KeyError: If ``cam_name`` is not present in the YAML.
            NotImplementedError: If the distortion model is not supported.
        """

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if cam_name not in data:
            raise KeyError(f"Camera '{cam_name}' not found in {yaml_path}.")

        cam = data[cam_name]

        fu, fv, cu, cv = cam['intrinsics']
        width, height = cam['resolution']
        D = np.array(cam['distortion_coeffs'], dtype=np.float64)

        distortion_model = CameraData.DistortionModel.from_kalibr_str(cam['distortion_model'])

        K = np.array([[fu, 0.0, cu],
                      [0.0, fv, cv],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = K

        return cls(frame_id=cam_name, width=int(width), height=int(height),
                   distortion_model=distortion_model,
                   K=K, D=D, R=R, P=P)

    @classmethod
    def from_ros1_bag(cls, bag_path: Union[Path, str], camera_info_topic: str) -> CameraData:
        """
        Load a CameraData instance from a ``sensor_msgs/CameraInfo`` topic in a
        ROS1 ``.bag`` file.

        Only the first message on ``camera_info_topic`` is read; camera
        calibration is assumed to be static across the bag.

        Args:
            bag_path: Path to the ``.bag`` file.
            camera_info_topic: Topic name of the ``sensor_msgs/CameraInfo``
                stream.

        Returns:
            CameraData: Instance populated with the calibration from the first
            message.

        Raises:
            ValueError: If ``camera_info_topic`` is not present in the bag or
                the bag contains no messages on that topic.
            NotImplementedError: If the distortion model in the message is not
                supported.
        """
        typestore = get_typestore(Stores.ROS1_NOETIC)

        with Reader1(Path(bag_path)) as reader:
            conns = [c for c in reader.connections if c.topic == camera_info_topic]
            if not conns:
                raise ValueError(
                    f"Topic {camera_info_topic!r} not found in bag {bag_path}.")
            conn = conns[0]

            msg = None
            for _, _, rawdata in reader.messages(connections=conns):
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                break

            if msg is None:
                raise ValueError(
                    f"No messages found on topic {camera_info_topic!r} in bag {bag_path}.")

            frame_id = msg.header.frame_id
            width = int(msg.width)
            height = int(msg.height)
            distortion_model = CameraData.DistortionModel.from_ros_str(msg.distortion_model)
            K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.D, dtype=np.float64).flatten()
            R = np.array(msg.R, dtype=np.float64).reshape(3, 3)
            P = np.array(msg.P, dtype=np.float64).reshape(3, 4)

        return cls(frame_id=frame_id, width=width, height=height,
                   distortion_model=distortion_model, K=K, D=D, R=R, P=P)

    # =========================================================================
    # ============================ Visualization ==============================
    # =========================================================================

    def visualize_FOV(self, depth: float = 5.0,
                      lidar_v_fov: Optional[Tuple[float, float]] = None,
                      testing: bool = False) -> None:
        """
        Visualize the camera field of view (FOV) as a 3D frustum, with an
        optional LiDAR FOV overlay.

        The camera is assumed to point along the +Z axis (standard optical
        convention), with X pointing right and Y pointing down. The frustum
        is derived from the intrinsic matrix ``K`` and the image dimensions.

        The optional LiDAR overlay assumes a full 360-degree horizontal FOV
        centred at the same origin, with a user-specified vertical angular
        range measured from the horizontal plane (positive angles are above
        the plane, negative angles are below).

        Args:
            depth: Depth in metres at which to draw the far face of the
                camera frustum and the LiDAR FOV rings.
            lidar_v_fov: If provided, a ``(min_deg, max_deg)`` tuple giving
                the vertical angle range of the LiDAR sensor in degrees.
                The LiDAR is assumed to cover a full 360-degree horizontal
                FOV and is drawn in the same scene.
            testing: If ``True``, suppresses ``plt.show()`` (used in unit
                tests).
        """

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Back-project the four image corners to 3D rays at the given depth.
        # Corner order: top-left, top-right, bottom-right, bottom-left.
        corners_px = np.array([
            [0,           0           ],
            [self.width,  0           ],
            [self.width,  self.height ],
            [0,           self.height ],
        ], dtype=np.float64)
        corners_3d = np.zeros((4, 3))
        corners_3d[:, 0] = (corners_px[:, 0] - cx) * depth / fx   # X (right)
        corners_3d[:, 1] = (corners_px[:, 1] - cy) * depth / fy   # Y (down)
        corners_3d[:, 2] = depth                                    # Z (forward)

        # Rotate so the optical axis (+Z camera) points along +Y world,
        # camera +X stays as world +X, and camera +Y maps to world -Z.
        R_cam_to_world = np.array([[1,  0,  0],
                                   [0,  0,  1],
                                   [0, -1,  0]], dtype=np.float64)
        corners_3d = (R_cam_to_world @ corners_3d.T).T

        origin = np.array([0.0, 0.0, 0.0])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Draw the four frustum edges from the origin to each far corner.
        for c in corners_3d:
            ax.plot([origin[0], c[0]], [origin[1], c[1]], [origin[2], c[2]],
                    color='steelblue', linewidth=1.5)

        # Draw the far-face rectangle.
        far_face_loop = np.vstack([corners_3d, corners_3d[0]])
        ax.plot(far_face_loop[:, 0], far_face_loop[:, 1], far_face_loop[:, 2],
                color='steelblue', linewidth=1.5)

        # Shade the five frustum faces (four side triangles + far rectangle).
        side_faces = [
            [origin.tolist(), corners_3d[0].tolist(), corners_3d[1].tolist()],
            [origin.tolist(), corners_3d[1].tolist(), corners_3d[2].tolist()],
            [origin.tolist(), corners_3d[2].tolist(), corners_3d[3].tolist()],
            [origin.tolist(), corners_3d[3].tolist(), corners_3d[0].tolist()],
            corners_3d.tolist(),
        ]
        frustum_poly = Poly3DCollection(
            side_faces, alpha=0.12, facecolor='steelblue', edgecolor='none')
        ax.add_collection3d(frustum_poly)

        ax.scatter(*origin, color='steelblue', s=60, zorder=5)

        h_fov_deg = np.degrees(2.0 * np.arctan(self.width  / (2.0 * fx)))
        v_fov_deg = np.degrees(2.0 * np.arctan(self.height / (2.0 * fy)))

        # --- Optional LiDAR FOV overlay ---
        if lidar_v_fov is not None:
            v_min_rad = np.radians(lidar_v_fov[0])
            v_max_rad = np.radians(lidar_v_fov[1])

            # Radius and height of the ring at each vertical extreme.
            # Points are at constant Euclidean distance `depth` from the origin.
            r_min = depth * np.cos(v_min_rad)
            r_max = depth * np.cos(v_max_rad)
            z_min = depth * np.sin(v_min_rad)
            z_max = depth * np.sin(v_max_rad)

            theta = np.linspace(0.0, 2.0 * np.pi, 200)

            # Two horizontal rings at the vertical FOV boundaries.
            ax.plot(r_min * np.cos(theta), r_min * np.sin(theta),
                    np.full_like(theta, z_min),
                    color='tomato', linewidth=1.5)
            ax.plot(r_max * np.cos(theta), r_max * np.sin(theta),
                    np.full_like(theta, z_max),
                    color='tomato', linewidth=1.5)

            # Straight spokes connecting the two rings.
            n_spokes = 24
            for phi in np.linspace(0.0, 2.0 * np.pi, n_spokes, endpoint=False):
                ax.plot(
                    [r_min * np.cos(phi), r_max * np.cos(phi)],
                    [r_min * np.sin(phi), r_max * np.sin(phi)],
                    [z_min, z_max],
                    color='tomato', linewidth=0.8, alpha=0.5)

        # --- Labels and legend ---
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Sensor Field of View')

        # Force equal scaling on all three axes by computing the max range across
        # all plotted data and applying it symmetrically around each axis centre.
        all_pts = np.vstack(corners_3d)
        if lidar_v_fov is not None:
            ring_pts = np.array([
                [depth * np.cos(v_min_rad), 0.0, depth * np.sin(v_min_rad)],
                [depth * np.cos(v_max_rad), 0.0, depth * np.sin(v_max_rad)],
            ])
            all_pts = np.vstack([all_pts, ring_pts])
        half_range = np.max(np.abs(all_pts)) * 1.05
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(-half_range, half_range)
        ax.set_zlim(-half_range, half_range)
        ax.set_box_aspect([1, 1, 1])

        legend_handles = [
            Line2D([0], [0], color='steelblue', linewidth=2,
                   label=f'Camera FOV  (H={h_fov_deg:.1f}°, V={v_fov_deg:.1f}°)')
        ]
        if lidar_v_fov is not None:
            legend_handles.append(
                Line2D([0], [0], color='tomato', linewidth=2,
                       label=(f'LiDAR FOV  (360° H, '
                              f'V=[{lidar_v_fov[0]:.1f}°, {lidar_v_fov[1]:.1f}°])')))
        ax.legend(handles=legend_handles, loc='upper left')

        if not testing:
            plt.show()

    # =========================================================================
    # =========================== Conversion to ROS ===========================
    # =========================================================================

    @staticmethod
    def get_ros_msg_type(lib_type: ROSMsgLibType) -> Any:
        """
        Return the ROS message type class for a CameraInfo message.

        Args:
            lib_type: Which ROS message library to use.

        Returns:
            The ROS message type class for ``sensor_msgs/CameraInfo``.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            return typestore.types['sensor_msgs/msg/CameraInfo'].__msgtype__
        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            return ModuleImporter.get_module_attribute('sensor_msgs.msg', 'CameraInfo')
        else:
            raise NotImplementedError(
                f"Unsupported ROSMsgLibType {lib_type} for CameraData.get_ros_msg_type()!")

    def get_ros_msg(self, lib_type: ROSMsgLibType, i: int = 0):
        """
        Build a CameraInfo ROS message from this calibration data using the
        timestamp stored at index ``i``.

        Args:
            lib_type: Which ROS message library to use.
            i: Index into the timestamps array (default 0).

        Returns:
            A populated ``sensor_msgs/CameraInfo`` message object.

        Raises:
            NotImplementedError: If ``lib_type`` is not supported.
        """

        seconds = int(self.timestamps[i])
        nanoseconds = int((self.timestamps[i] - self.timestamps[i].to_integral_value(
            rounding=decimal.ROUND_DOWN)) * Decimal("1e9").to_integral_value(decimal.ROUND_HALF_EVEN))

        distortion_str = CameraData.DistortionModel.to_ros_str(self.distortion_model)

        if lib_type == ROSMsgLibType.ROSBAGS:
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
            Header = typestore.types['std_msgs/msg/Header']
            Time = typestore.types['builtin_interfaces/msg/Time']
            RegionOfInterest = typestore.types['sensor_msgs/msg/RegionOfInterest']

            return CameraInfo(
                header=Header(
                    stamp=Time(sec=seconds, nanosec=nanoseconds),
                    frame_id=self.frame_id),
                height=self.height,
                width=self.width,
                distortion_model=distortion_str,
                d=self.D.copy(),
                k=self.K.flatten().copy(),
                r=self.R.flatten().copy(),
                p=self.P.flatten().copy(),
                binning_x=0,
                binning_y=0,
                roi=RegionOfInterest(
                    x_offset=0, y_offset=0,
                    height=0, width=0,
                    do_rectify=False))

        elif lib_type == ROSMsgLibType.RCLPY or lib_type == ROSMsgLibType.ROSPY:
            Header = ModuleImporter.get_module_attribute('std_msgs.msg', 'Header')
            CameraInfo = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'CameraInfo')
            RegionOfInterest = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'RegionOfInterest')

            msg = CameraInfo()
            msg.header = Header()
            msg.header.frame_id = self.frame_id

            if lib_type == ROSMsgLibType.RCLPY:
                Time = ModuleImporter.get_module_attribute('rclpy.time', 'Time')
                msg.header.stamp = Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()
            else:
                rospy = ModuleImporter.get_module('rospy')
                msg.header.stamp = rospy.Time(secs=seconds, nsecs=nanoseconds)

            msg.height = self.height
            msg.width = self.width
            msg.distortion_model = distortion_str
            msg.binning_x = 0
            msg.binning_y = 0
            roi = RegionOfInterest()
            roi.x_offset = 0
            roi.y_offset = 0
            roi.height = 0
            roi.width = 0
            roi.do_rectify = False
            msg.roi = roi

            # ROS1 uses uppercase field names; ROS2 uses lowercase
            if lib_type == ROSMsgLibType.ROSPY:
                msg.D = self.D.tolist()
                msg.K = self.K.flatten().tolist()
                msg.R = self.R.flatten().tolist()
                msg.P = self.P.flatten().tolist()
            else:
                msg.d = self.D.tolist()
                msg.k = self.K.flatten().tolist()
                msg.r = self.R.flatten().tolist()
                msg.p = self.P.flatten().tolist()
            return msg

        else:
            raise NotImplementedError(
                f"Unsupported ROSMsgLibType {lib_type} for CameraData.get_ros_msg()!")

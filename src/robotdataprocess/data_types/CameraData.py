from __future__ import annotations

from .Data import ROSMsgLibType
from .SequentialData import SequentialData
import decimal
from decimal import Decimal
from enum import Enum
from ..ModuleImporter import ModuleImporter
import numpy as np
from numpy.typing import NDArray
from rosbags.typesys import Stores, get_typestore
from typeguard import typechecked
from typing import Any, Union
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

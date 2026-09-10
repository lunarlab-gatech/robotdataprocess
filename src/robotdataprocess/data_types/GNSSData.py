from ..utils.conversion_utils import col_to_dec_arr
from .SequentialData import SequentialData
from decimal import Decimal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from rosbags.rosbag1 import Reader as Reader1
from rosbags.typesys import Stores, get_typestore
from typeguard import typechecked
from typing import Union
import tqdm

@typechecked
class GNSSData(SequentialData):
    """
    GNSS fix data with latitude/longitude/altitude, position covariance, and fix status.

    Supports loading from ROS1 bags.

    Attributes:
        lat_lon_alt: (N, 3) array of latitude (deg), longitude (deg), and altitude (m).
        position_covariance: (N, 3, 3) array of position covariance matrices, in
            (east, north, up) or (x, y, z) coordinates as specified by the source message.
        position_covariance_type: (N,) array of covariance type constants, matching
            ``sensor_msgs/NavSatFix.position_covariance_type``.
        status: (N,) array of fix status constants, matching ``sensor_msgs/NavSatStatus.status``.
        service: (N,) array of GNSS service constants, matching ``sensor_msgs/NavSatStatus.service``.
    """

    # Define GNSS-specific data attributes
    lat_lon_alt: NDArray
    position_covariance: NDArray
    position_covariance_type: NDArray
    status: NDArray
    service: NDArray

    @typechecked
    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, list],
                 lat_lon_alt: Union[np.ndarray, list], position_covariance: Union[np.ndarray, list],
                 position_covariance_type: Union[np.ndarray, list],
                 status: Union[np.ndarray, list], service: Union[np.ndarray, list]):

        # Copy initial values into attributes
        super().__init__(frame_id, timestamps)
        self.lat_lon_alt = col_to_dec_arr(lat_lon_alt)
        self.position_covariance = col_to_dec_arr(position_covariance)
        self.position_covariance_type = np.asarray(position_covariance_type)
        self.status = np.asarray(status)
        self.service = np.asarray(service)

        # Check to ensure that all arrays have same length
        if len(self.timestamps) != len(self.lat_lon_alt) or len(self.lat_lon_alt) != len(self.position_covariance) \
            or len(self.position_covariance) != len(self.position_covariance_type) \
            or len(self.position_covariance_type) != len(self.status) or len(self.status) != len(self.service):
            raise ValueError("Lengths of timestamp, lat_lon_alt, position_covariance, position_covariance_type, "
                              "status, and service arrays are not equal!")

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in GNSSData. """
        pass

    def __eq__(self, other) -> bool:
        parent_result = super().__eq__(other)
        if parent_result is not True:
            return parent_result
        if not np.array_equal(self.lat_lon_alt, other.lat_lon_alt):
            print(f"  [__eq__] lat_lon_alt not equal")
            return False
        if not np.array_equal(self.position_covariance, other.position_covariance):
            print(f"  [__eq__] position_covariance not equal")
            return False
        if not np.array_equal(self.position_covariance_type, other.position_covariance_type):
            print(f"  [__eq__] position_covariance_type not equal")
            return False
        if not np.array_equal(self.status, other.status):
            print(f"  [__eq__] status not equal")
            return False
        if not np.array_equal(self.service, other.service):
            print(f"  [__eq__] service not equal")
            return False
        return True

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    @typechecked
    def from_ros1_bag(cls, bag_path: Union[Path, str], gnss_topic: str):
        """
        Creates a class structure from a ROS1 bag file with a NavSatFix topic.

        Args:
            bag_path (Path | str): Path to the ROS1 .bag file.
            gnss_topic (str): Topic of the sensor_msgs/NavSatFix messages.
        Returns:
            GNSSData: Instance of this class.
        Raises:
            ValueError: If ``gnss_topic`` is not present in the bag.
        """

        typestore = get_typestore(Stores.ROS1_NOETIC)

        with Reader1(Path(bag_path)) as reader:
            conns = [c for c in reader.connections if c.topic == gnss_topic]
            if not conns:
                raise ValueError(f"Topic {gnss_topic!r} not found in bag {bag_path}.")
            conn = conns[0]

            num_msgs = len(reader.indexes[conn.id])
            timestamps_np = np.zeros(num_msgs, dtype=Decimal)
            lat_lon_alt_np = np.zeros((num_msgs, 3), dtype=Decimal)
            position_covariance_np = np.zeros((num_msgs, 3, 3), dtype=Decimal)
            position_covariance_type_np = np.zeros(num_msgs, dtype=np.uint8)
            status_np = np.zeros(num_msgs, dtype=np.int8)
            service_np = np.zeros(num_msgs, dtype=np.uint16)

            frame_id = None
            pbar = tqdm.tqdm(total=num_msgs, desc="Extracting GNSS...", unit=" msgs")

            for i, (_, _, rawdata) in enumerate(reader.messages(connections=conns)):
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

                if i == 0:
                    frame_id = msg.header.frame_id

                timestamps_np[i] = (Decimal(msg.header.stamp.sec) +
                                    Decimal(msg.header.stamp.nanosec) * Decimal('1e-9'))
                lat_lon_alt_np[i] = np.array([Decimal(str(msg.latitude)), Decimal(str(msg.longitude)),
                                              Decimal(str(msg.altitude))])
                position_covariance_np[i] = col_to_dec_arr(np.array(msg.position_covariance)).reshape(3, 3)
                position_covariance_type_np[i] = msg.position_covariance_type
                status_np[i] = msg.status.status
                service_np[i] = msg.status.service

                pbar.update(1)

        return cls(frame_id, timestamps_np, lat_lon_alt_np, position_covariance_np,
                   position_covariance_type_np, status_np, service_np)

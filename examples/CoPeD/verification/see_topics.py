from getpass import getuser
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from robotdataprocess import OdometryData, CoordinateFrame, ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.conversion_utils import dec_arr_to_float_arr
from rosbags.highlevel import AnyReader

def main():

    sequence: str = "HOUSEB"
    robot_name: str = "race1"
    bag_path: Path = Path('/media') / getuser() / 'T73' / 'CoPeD_dataset' / sequence / 'data' / f'{sequence}_{robot_name}_2023_05_19_04_00_PM_1.bag'

    with AnyReader([bag_path]) as reader:
        for connection in reader.connections:
            print(connection.topic, connection.msgtype, connection.msgcount)

    """
    /race1/cam1/imu sensor_msgs/msg/Imu 88176
    /race1/cam1/color/image_raw/compressed sensor_msgs/msg/CompressedImage 13252
    /race1/cam1/infra1/image_rect_raw/compressed sensor_msgs/msg/CompressedImage 13258
    /race1/cam1/infra2/image_rect_raw/compressed sensor_msgs/msg/CompressedImage 13258
    /race1/mavros/global_position/raw/fix sensor_msgs/msg/NavSatFix 3537
    /race1/main_camera/image_raw/compressed sensor_msgs/msg/CompressedImage 13258
    /race1/mavros/imu/data sensor_msgs/msg/Imu 22095
    /race1/mavros/local_position/odom nav_msgs/msg/Odometry 13256
    /race1/mavros/global_position/raw/gps_vel geometry_msgs/msg/TwistStamped 3536
    /race1/mavros/gpsstatus/gps1/raw mavros_msgs/msg/GPSRAW 3536
    """

    odom_data = OdometryData.from_ros1_bag(bag_path, f'/{robot_name}/mavros/local_position/odom', CoordinateFrame.NONE)
    image_data = ImageDataOnDisk.from_ros1_bag(bag_path, f'/{robot_name}/cam1/color/image_raw/compressed')
    print(f"Encoding: {image_data.encoding}")

    odom_data.visualize_3D([], ["odom"])

    odom_timestamps = dec_arr_to_float_arr(odom_data.timestamps)
    odom_positions = dec_arr_to_float_arr(odom_data.positions)
    odom_orientations = dec_arr_to_float_arr(odom_data.orientations)

    fig, (ax_image, ax_odom) = plt.subplots(1, 2, figsize=(12, 6))

    im = ax_image.imshow(_to_rgb(image_data, 0))
    ax_image.set_title(f"Frame 0/{len(image_data.images)}")

    odom_line, = ax_odom.plot([], [])
    ax_odom.set_xlim(odom_positions[:, 0].min(), odom_positions[:, 0].max())
    ax_odom.set_ylim(odom_positions[:, 1].min(), odom_positions[:, 1].max())
    ax_odom.set_xlabel("x [m]")
    ax_odom.set_ylabel("y [m]")
    ax_odom.set_aspect("equal")

    arrow_length = 0.05 * max(np.ptp(odom_positions[:, 0]), np.ptp(odom_positions[:, 1]))
    heading_quiver = ax_odom.quiver(0, 0, 1, 0, color="r", scale_units="xy", angles="xy", scale=1)

    playback_speed: int = 1
    skip_seconds: float = 200.0

    start_time = float(image_data.timestamps[0]) + skip_seconds
    start_i = int(np.searchsorted(dec_arr_to_float_arr(image_data.timestamps), start_time, side="left"))

    for i in range(start_i, len(image_data.images), playback_speed):
        im.set_data(_to_rgb(image_data, i))
        ax_image.set_title(f"Frame {i}/{len(image_data.images)}")

        image_timestamp = float(image_data.timestamps[i])
        idx = np.searchsorted(odom_timestamps, image_timestamp, side="right")
        odom_line.set_data(odom_positions[:idx, 0], odom_positions[:idx, 1])

        last_position = odom_positions[idx - 1]
        heading = R.from_quat(odom_orientations[idx - 1]).apply([1, 0, 0])
        heading_quiver.set_offsets([[last_position[0], last_position[1]]])
        heading_quiver.set_UVC(heading[0] * arrow_length, heading[1] * arrow_length)

        plt.pause(0.001)

    plt.show()

def _to_rgb(image_data: ImageDataOnDisk, i: int):
    image = image_data.images[i]
    if image_data.encoding == ImageData.ImageEncoding.BGR8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    main()

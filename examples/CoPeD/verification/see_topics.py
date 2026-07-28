from getpass import getuser
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from robotdataprocess import OdometryData, CoordinateFrame, ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.utils.conversion_utils import dec_arr_to_float_arr
from rosbags.highlevel import AnyReader

def main():

    sequence: str = "HOUSEA"
    robot_name: str = "wanda"
    bag_path: Path = Path('/media') / getuser() / 'T73' / 'CoPeD_dataset' / sequence / 'data' / f'{sequence}_{robot_name}_arl_outtdoor_2023_05_19_04_2023-05-19-15-37-54.bag'

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

    """
    /tf tf2_msgs/msg/TFMessage 25814
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 71921
    /rosout_agg rosgraph_msgs/msg/Log 6531
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 71842
    /tf_static tf2_msgs/msg/TFMessage 1
    /tf tf2_msgs/msg/TFMessage 61894
    /tf_static tf2_msgs/msg/TFMessage 1
    /wilbur/accepted_frame_notifications omnimapper_msgs/msg/FrameNotificationSymbol 278
    /tf_static tf2_msgs/msg/TFMessage 1
    /tf_static tf2_msgs/msg/TFMessage 1
    /wilbur/forward/color/camera_info sensor_msgs/msg/CameraInfo 14892
    /tf_aggregate tf2_msgs/msg/TFMessage 4968
    /wilbur/cmd_vel geometry_msgs/msg/Twist 14902
    /wilbur/forward/aligned_depth_to_color/image_raw/compressed sensor_msgs/msg/CompressedImage 14867
    /wilbur/forward/color/image_rect_color/compressed sensor_msgs/msg/CompressedImage 14892
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 8522
    /tf tf2_msgs/msg/TFMessage 4966
    /wilbur/forward/extrinsics/depth_to_color realsense2_camera/msg/Extrinsics 1
    /wilbur/forward/depth/camera_info sensor_msgs/msg/CameraInfo 14918
    /wilbur/frame_notifications omnimapper_msgs/msg/FrameNotifications 288
    /wilbur/forward/infra2/camera_info sensor_msgs/msg/CameraInfo 14890
    /wilbur/forward/infra1/camera_info sensor_msgs/msg/CameraInfo 14890
    /wilbur/forward/depth/image_rect_raw/compressed sensor_msgs/msg/CompressedImage 14917
    /wilbur/forward/infra2/image_rect_raw/compressed sensor_msgs/msg/CompressedImage 14890
    /wilbur/cmd_lights warthog_msgs/msg/Lights 2483
    /wilbur/forward/infra1/image_rect_raw/compressed sensor_msgs/msg/CompressedImage 14890
    /diagnostics_toplevel_state diagnostic_msgs/msg/DiagnosticStatus 497
    /wilbur/imu/data sensor_msgs/msg/Imu 62057
    /diagnostics_agg diagnostic_msgs/msg/DiagnosticArray 497
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 497
    /wilbur/imu/filter/status arl_sensor_msgs/msg/ImuStatus 62051
    /wilbur/joint_states sensor_msgs/msg/JointState 34743
    /wilbur/left_drive/velocity std_msgs/msg/Float64 9925
    /wilbur/master_discovery/changes fkie_multimaster_msgs/msg/MasterState 4
    /wilbur/odom nav_msgs/msg/Odometry 62009
    /wilbur/mcu/enable_motors std_msgs/msg/Bool 4962
    /wilbur/imu/mag sensor_msgs/msg/MagneticField 14777
    /tf_static tf2_msgs/msg/TFMessage 1
    /wilbur/lidar_points_front sensor_msgs/msg/PointCloud2 4963
    /wilbur/lidar_points_center sensor_msgs/msg/PointCloud2 4963
    /wilbur/lidar_points sensor_msgs/msg/PointCloud2 4963
    /wilbur/pose_graph omnimapper_msgs/msg/PoseGraph 278
    /wilbur/pose geometry_msgs/msg/PoseStamped 61995
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 316
    /wilbur/mcu/cmd_lights warthog_msgs/msg/Lights 2481
    /wilbur/right_drive/velocity std_msgs/msg/Float64 9918
    /wilbur/stereo_left/camera_info sensor_msgs/msg/CameraInfo 14883
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 2954
    /wilbur/stereo_left/image_rect_color/compressed sensor_msgs/msg/CompressedImage 14882
    /wilbur/local_point_cloud_cache/renderers/recent_map_compressed zip/msg/CompressedMessage 4960
    /wilbur/worldmodel_rviz/object_markers visualization_msgs/msg/MarkerArray 497
    /wilbur/warthog_velocity_controller/odom nav_msgs/msg/Odometry 9911
    /wilbur/warthog_velocity_controller/cmd_vel_out geometry_msgs/msg/TwistStamped 9915
    /wilbur/stereo_right/camera_info sensor_msgs/msg/CameraInfo 14880
    /wilbur/warthog_velocity_controller/cmd_vel geometry_msgs/msg/Twist 7455
    /wilbur/point_cloud_cache/renderers/full_map_compressed zip/msg/CompressedMessage 418
    /wilbur/master_discovery/linkstats fkie_multimaster_msgs/msg/LinkStatesStamped 496
    /wilbur/stereo_right/image_rect_color/compressed sensor_msgs/msg/CompressedImage 14877
    /diagnostics_agg diagnostic_msgs/msg/DiagnosticArray 496
    /diagnostics_toplevel_state diagnostic_msgs/msg/DiagnosticStatus 496
    /wilbur/mcu/status warthog_msgs/msg/Status 437
    /diagnostics diagnostic_msgs/msg/DiagnosticArray 436
    /wilbur/rc_teleop/cmd_vel geometry_msgs/msg/Twist 6308
    /wilbur/right_drive/status/speed std_msgs/msg/Float64 4309
    /wilbur/right_drive/status/fault std_msgs/msg/Bool 1723
    /wilbur/left_drive/status/speed std_msgs/msg/Float64 1722
    /wilbur/left_drive/status/fault std_msgs/msg/Bool 1722
    /wilbur/right_drive/status/battery_current std_msgs/msg/Float64 430
    /wilbur/right_drive/status/battery_voltage std_msgs/msg/Float64 430
    /wilbur/right_drive/status/motor_temperature std_msgs/msg/Int32 430
    /wilbur/left_drive/status/battery_current std_msgs/msg/Float64 429
    /wilbur/left_drive/status/battery_voltage std_msgs/msg/Float64 429
    /wilbur/left_drive/status/motor_temperature std_msgs/msg/Int32 429
    """

    # odom_data = OdometryData.from_ros1_bag(bag_path, f'/{robot_name}/mavros/local_position/odom', CoordinateFrame.NONE)
    odom_data = OdometryData.from_ros1_bag(bag_path, f'/{robot_name}/pose', CoordinateFrame.NONE, "PoseStamped", "robot")
    #image_data = ImageDataOnDisk.from_ros1_bag(bag_path, f'/{robot_name}/cam1/color/image_raw/compressed')
    #print(f"Encoding: {image_data.encoding}")

    odom_data.visualize_3D([], ["odom"], axes_length=3)

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

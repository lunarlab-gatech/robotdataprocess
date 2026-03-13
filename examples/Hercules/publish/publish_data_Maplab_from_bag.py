import argparse
from decimal import Decimal
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union
import numpy as np

# =============================================================================
# Edit topic names and configuration here
# =============================================================================
IMU_TOPIC        = '/gnss/imu'
ODOM_TOPIC       = '/gnss/ground_truth'
LEFT_IMG_TOPIC   = '/camera_left/image_raw'
RIGHT_IMG_TOPIC  = '/camera_right/image_raw'
LIDAR_TOPIC      = '/velodyne/points'

# Transformation from LiDAR frame (L) to IMU frame (I), expressed in NED.
# Used only when robot_name is not 'Husky' or 'Drone'.
# LIO-SAM outputs W->L poses; Maplab needs W->I = W->L @ H_L_TO_I.
# Set the translation column [x, y, z] to the IMU origin in the LiDAR frame.
# Example: z=0.85 means the IMU is 0.85 m below the LiDAR along NED Z (down).
H_L_TO_I = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
# =============================================================================


def publish_data(bag_path: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")

    # Load all data from the ROS1 bag
    imu_data = ImuData.from_ros1_bag(bag_path, IMU_TOPIC,
                                     robot_name + '/base_link', CoordinateFrame.NED)

    odom_data = OdometryData.from_ros1_bag(bag_path, ODOM_TOPIC, CoordinateFrame.NED)

    # Get L->I transformation
    if "Husky" in robot_name:
        H_L_to_I_in_NED = np.array([[1.0,  0.0,  0.0,  0.0],
                                    [0.0,  1.0,  0.0,  0.0],
                                    [0.0,  0.0,  1.0, 0.85],
                                    [0.0,  0.0,  0.0,  1.0]])
    elif "Drone" in robot_name:
        H_L_to_I_in_NED = np.array([[1.0,  0.0,  0.0,  0.0],
                                    [0.0,  1.0,  0.0,  0.0],
                                    [0.0,  0.0,  1.0,  0.5],
                                    [0.0,  0.0,  0.0,  1.0]])
    else:
        H_L_to_I_in_NED = H_L_TO_I

    # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)
    odom_data.apply_transformation_right_side(H_L_to_I_in_NED)

    left_image_data  = ImageDataInMemory.from_ros1_bag(bag_path, LEFT_IMG_TOPIC)
    right_image_data = ImageDataInMemory.from_ros1_bag(bag_path, RIGHT_IMG_TOPIC)
    lidar_data = LiDARData.from_ros1_bag(bag_path, LIDAR_TOPIC,
                                         robot_name + '/lidar_link', CoordinateFrame.NED)

    lidar_data.calculate_point_channels(16, -20, 20)

    # Crop the data
    if crop_data:
        imu_data.crop_data(Decimal('0.0'), end_time)
        odom_data.crop_data(Decimal('0.0'), end_time)
        left_image_data.crop_data(Decimal('0.0'), end_time)
        right_image_data.crop_data(Decimal('0.0'), end_time)
        raise NotImplementedError("Crop Data not implemented for lidar_data!")

    # Publish the data via ROS1 topics
    publish_data_ROS_multiprocess([imu_data, left_image_data, right_image_data, odom_data, odom_data, lidar_data],
                                  ['/' + robot_name + '/gnss/imu',
                                   '/' + robot_name + '/ground1/camera_left',
                                   '/' + robot_name + '/ground1/camera_right',
                                   '/' + robot_name + '/odom_gt',
                                   '/' + robot_name + '/rovioli/maplab_odom_T_M_I',
                                   '/' + robot_name + '/lidar'],
                                   [None, None, None, "Path", "maplab_msg/OdometryWithImuBiases", None],
                                   [200, 20, 20, 20, 20, 20],
                                   [1, 1, 1, 1, 1, 1], ROSMsgLibType.ROSPY, True, verbose=True)


def main(bag_path: str, robot_name: str, crop_end_time: Union[float, None]):
    if crop_end_time is None:
        crop_data = False
        crop_end_time = None
    else:
        crop_data = True
        crop_end_time = Decimal(crop_end_time)

    publish_data(bag_path=bag_path,
                 robot_name=robot_name,
                 crop_data=crop_data,
                 end_time=crop_end_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Hercules data from a ROS1 bag via ROS1 topics for Maplab.")
    parser.add_argument('--bag_path', type=str, required=True, help="Path to the ROS1 .bag file.")
    parser.add_argument('--robot_name', type=str, required=True, help="Name of the robot (e.g., Husky1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(bag_path=args.bag_path, robot_name=args.robot_name, crop_end_time=args.crop_end_time)

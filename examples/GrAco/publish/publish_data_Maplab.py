import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union
import numpy as np

H_L_TO_I = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])

def publish_data(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Extract RGB and IMU from Hercules
    input_path = Path(input_dir).absolute() 
    imu_data = ImuData.from_txt_file(input_path / robot_name / 'imu' / 'imu.txt', '' + robot_name + '/base_link', CoordinateFrame.NED)
    odom_data = OdometryData.from_csv(input_path / robot_name / 'odom' / 'odom.csv', robot_name + '/odom', robot_name + '/ground_truth/base_link', CoordinateFrame.NED, True)
    # odom.csv is 200Hz; downsample to 20Hz by keeping every 10th sample
    step = round(len(odom_data.timestamps) / (float(odom_data.timestamps[-1] - odom_data.timestamps[0]) * 20))
    odom_data.timestamps = odom_data.timestamps[::step]
    odom_data.positions = odom_data.positions[::step]
    odom_data.orientations = odom_data.orientations[::step]
    odom_data.poses = []
    odom_data.poses_rclpy = []
    # odom_data.to_FLU_frame()

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

    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'camera_left' / 'images', '' + robot_name + '/front_center_Scene')
    right_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'camera_right' / 'images', '' + robot_name + '/front_right_Scene')
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", "lidar_link", CoordinateFrame.NED)
    if lidar_data is None:
        lidar_data.calculate_point_channels(16, -20, 20)

    # Crop the data
    if crop_data:
        imu_data.crop_data(Decimal('0.0'), end_time)
        odom_data.crop_data(Decimal('0.0'), end_time)
        left_image_data.crop_data(Decimal('0.0'), end_time)
        right_image_data.crop_data(Decimal('0.0'), end_time)
        raise NotADirectoryError("Crop Data not implemented for lidar_data!")

    # Publish the data via ROS2 topics
    publish_data_ROS_multiprocess([imu_data, left_image_data, right_image_data, odom_data, odom_data, lidar_data], 
                                  ['/'+robot_name+'/imu0',
                                   '/'+robot_name+'/cam0/image_raw',
                                   '/'+robot_name+'/cam1/image_raw',
                                   '/'+robot_name+'/odom_gt',
                                   '/'+robot_name+'/rovioli/maplab_odom_T_M_I',
                                   '/'+robot_name+'/lidar'],
                                    [None, None, None, "Path", "maplab_msg/OdometryWithImuBiases", None], 
                                    [200, 20, 20, 20, 20, 20],
                                    [1, 1, 1, 1, 1, 1], ROSMsgLibType.ROSPY, True, verbose=True)
    # publish_data_ROS_multiprocess([imu_data, left_image_data, right_image_data, odom_data], 
    #                               ['/'+robot_name+'/imu0', 
    #                                '/'+robot_name+'/cam0/image_raw', 
    #                                '/'+robot_name+'/cam1/image_raw',
    #                                '/'+robot_name+'/rovioli/maplab_odom_T_M_I',
    #                                '/'+robot_name+'/odom_gt'],
    #                                 [None, None, None, "maplab_msg/OdometryWithImuBiases", "Path"], 
    #                                 [1, 3, 3, 1, 1], ROSMsgLibType.ROSPY, True, verbose=True)

def main(dataset_num: str, robot_name: str, crop_end_time: Union[float, None]): 

    # Publish data for each robot
    if crop_end_time == None: 
        crop_data = False
        crop_end_time = None
    else: 
        crop_data = True
        crop_end_time = Decimal(crop_end_time)

    user = getpass.getuser()
    publish_data(input_dir='/home/' + user + '/data/Data/GrAco_dataset/' + dataset_num,
                    robot_name=robot_name,
                    crop_data=crop_data,
                    end_time=crop_end_time)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish GrAco data via ROS2 topics for Maplab.")
    parser.add_argument('--dataset_num', type=str, default=None, help="Dataset version number (e.g., V1.6).")
    parser.add_argument('--robot_name', type=str, default=None, help="Name of the robot (e.g., Husky1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(dataset_num=args.dataset_num, robot_name=args.robot_name, crop_end_time=args.crop_end_time)
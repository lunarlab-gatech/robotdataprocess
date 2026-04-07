import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str, start_time: Decimal, end_time: Union[Decimal, None]):
    # Extract data from Hercules
    input_path = Path(input_dir).absolute() 
    imu_data = ImuData.from_txt(input_path / robot_name / 'synthetic_imu_9axis_500Hz.txt', '' + robot_name + '/base_link', CoordinateFrame.NED, True)
    pose_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', 'global', 'body', CoordinateFrame.NED, False)
    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', '' + robot_name + '/front_center_Scene')
    right_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_right', '' + robot_name + '/front_center_Scene')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    pose_data.to_FLU_frame()
    pose_data.shift_to_start_at_identity()

    # Crop the data
    imu_data.crop_data(start_time, end_time)
    pose_data.crop_data(start_time, end_time)
    left_image_data.crop_data(start_time, end_time)
    right_image_data.crop_data(start_time, end_time)

    # Publish the data via ROS2 topics 
    publish_data_ROS_multiprocess([imu_data, pose_data, left_image_data, right_image_data], 
                                  ['/imu0', '/odom_gt', '/cam0/image_raw', '/cam1/image_raw'],
                                  [None, "Path", None, None],
                                  [500, 20, 20, 20],
                                  [1, 1, 3, 3],
                                   ROSMsgLibType.RCLPY, True, verbose=True)
    
def main(dataset_num: str, robot_name: str, crop_start_time: float, crop_end_time: Union[float, None]): 

    # Do bookkeeping for cropping
    crop_start_time = Decimal(crop_start_time)
    crop_end_time = Decimal(crop_end_time) if crop_end_time is not None else None
    
    # Publish the data for the specified robot
    user = getpass.getuser()
    publish_data(input_dir='/home/' + user + '/data/Hercules_datasets/' + dataset_num + '/data',
                    robot_name=robot_name,
                    start_time=crop_start_time,
                    end_time=crop_end_time)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Hercules data via ROS2 topics for VINS-Mono.")
    parser.add_argument('--dataset_num', type=str, default=None, help="Dataset version number.")
    parser.add_argument('--robot_name', type=str, default=None, help="Name of the robot (e.g., Drone1).")
    args = parser.parse_args()

    # OpenVINS starts from zero as it desires static initialization
    if args.dataset_num == "V2.4.C":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('382.85'), Decimal('390.90'), Decimal('1100.00'), Decimal('1190.35')]
    elif args.dataset_num == "V2.3.AP":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('772.15'), Decimal('741.45'), Decimal('1121.80'), Decimal('1193.80')]
    elif args.dataset_num == "V2.3.AC":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('1125.00'), Decimal('1118.80'), Decimal('1025.50'), Decimal('892.60')]
    elif args.dataset_num == "V2.3.C":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('4105.00'), Decimal('4105.00'), Decimal('4105.00'), Decimal('4105.00')]
    elif args.dataset_num == "V2.4.F":
        robot_crop_start_times = [Decimal('35.05'), Decimal('34.60'), Decimal('27.45'), Decimal('31.50')]
        robot_crop_end_times = [Decimal('575.55'), Decimal('762.35'), Decimal('898.10'), Decimal('906.85')]
    else:
        raise ValueError("Crop times not specified for this dataset number.")
    
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

    # Get start and end times for the specified robot
    if args.robot_name not in robot_names:
        raise ValueError(f"Invalid robot name. Must be one of {robot_names}.")
    robot_index = robot_names.index(args.robot_name)
    crop_start_time = robot_crop_start_times[robot_index]
    crop_end_time = robot_crop_end_times[robot_index]


    main(dataset_num=args.dataset_num, robot_name=args.robot_name, crop_start_time=crop_start_time, crop_end_time=crop_end_time)
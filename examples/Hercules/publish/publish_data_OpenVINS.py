import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Extract data from Hercules
    input_path = Path(input_dir).absolute() 
    imu_data = ImuData.from_txt_file(input_path / robot_name / 'synthetic_imu_9axis_500Hz.txt', '' + robot_name + '/base_link', CoordinateFrame.NED, True)
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 'global', 'body', CoordinateFrame.NED, False)
    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', '' + robot_name + '/front_center_Scene')
    right_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_right', '' + robot_name + '/front_center_Scene')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    pose_data.to_FLU_frame()
    pose_data.shift_to_start_at_identity()

    # Crop the data
    if crop_data:
        imu_data.crop_data(Decimal('0.0'), end_time)
        pose_data.crop_data(Decimal('0.0'), end_time)
        left_image_data.crop_data(Decimal('0.0'), end_time)
        right_image_data.crop_data(Decimal('0.0'), end_time)

    # Publish the data via ROS2 topics 
    publish_data_ROS_multiprocess([imu_data, pose_data, left_image_data, right_image_data], 
                                  ['/imu0', '/odom_gt', '/cam0/image_raw', '/cam1/image_raw'],
                                  [None, "Path", None, None],
                                  [500, 20, 20, 20],
                                  [1, 1, 3, 3],
                                   ROSMsgLibType.RCLPY, True, verbose=True)
    
def main(dataset_num: str, robot_name: str, crop_end_time: Union[float, None]): 

    # Do bookkeeping for cropping
    if crop_end_time == None: 
        crop_data = False
        crop_end_time = None
    else: 
        crop_data = True
        crop_end_time = Decimal(crop_end_time)
    
    # Publish the data for the specified robot
    user = getpass.getuser()
    publish_data(input_dir='/home/' + user + '/data/Hercules_datasets/' + dataset_num + '/data',
                    robot_name=robot_name,
                    crop_data=crop_data,
                    end_time=crop_end_time)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Hercules data via ROS2 topics for VINS-Mono.")
    parser.add_argument('--dataset_num', type=str, default=None, help="Dataset version number (e.g., V1.6).")
    parser.add_argument('--robot_name', type=str, default=None, help="Name of the robot (e.g., Drone1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(dataset_num=args.dataset_num, robot_name=args.robot_name, crop_end_time=args.crop_end_time)
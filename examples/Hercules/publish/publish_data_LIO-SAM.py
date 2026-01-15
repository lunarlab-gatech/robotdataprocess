import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.LiDARData import LiDARData
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Extract data from Hercules
    input_path = Path(input_dir).absolute() 
    imu_data = ImuData.from_txt_file(input_path / robot_name / 'synthetic_imu_9axis.txt', 'base_link', 
                                     CoordinateFrame.NED, nine_axis=True)
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 'map', 'base_link', CoordinateFrame.NED, False)
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", "lidar_link", CoordinateFrame.NED)

    # Convert LiDAR data to FLU frame
    lidar_data.to_FLU_frame()
    lidar_data.calculate_point_channels(32, -25, 25)

    # Convert GT Pose to FLU frame as well
    pose_data.to_FLU_frame()
    pose_data.shift_to_start_at_identity()

    # Crop the data
    if crop_data:
        imu_data.crop_data(Decimal('0.0'), end_time)
        pose_data.crop_data(Decimal('0.0'), end_time)
        lidar_data.crop_data(Decimal('0.0'), end_time)

    # Publish the data via ROS topics 
    publish_data_ROS_multiprocess([imu_data, pose_data, lidar_data], 
                                  ['/imu_raw', '/odom_gt', '/points_raw'],
                                  [None, "Path", None],
                                  [1, 1, 1],
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
    parser = argparse.ArgumentParser(description="Publish Hercules data via ROS topics for LIO-SAM.")
    parser.add_argument('--dataset_num', type=str, default=None, required=True, help="Dataset version number (e.g., V1.6).")
    parser.add_argument('--robot_name', type=str, default=None, required=True, help="Name of the robot (e.g., Drone1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(dataset_num=args.dataset_num, robot_name=args.robot_name, crop_end_time=args.crop_end_time)
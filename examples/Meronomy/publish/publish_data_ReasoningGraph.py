import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str):

    # Extract RGB, Depth, and odometery
    input_path = Path(input_dir).absolute() 
    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', '' + robot_name + '/front_center_Scene')
    depth_data = ImageDataOnDisk.from_npy_files(input_path / robot_name / 'depth', 'front_center_DepthPerspective')
    odom_data = OdometryData.from_csv(input_path.parent / 'results' / 'LIO-SAM' / robot_name / 'odometry.csv', "world", "robot", CoordinateFrame.NED, True, None)  

    # Publish the data via ROS2 topics
    publish_data_ROS_multiprocess([left_image_data, depth_data, odom_data], 
                                  ['/'+robot_name+'/cam0/image_raw',
                                   '/'+robot_name+'/cam0/depth/image_raw',
                                   '/tf'],
                                    [None, None, "TFMessage"], 
                                    [20, 20, 20],
                                    [1, 1, 1], ROSMsgLibType.ROSPY, True, verbose=True)

def main(dataset_num: str, robot_name: str): 
    user = getpass.getuser()
    publish_data(input_dir='/home/' + user + '/data/Meronomy_datasets/' + dataset_num + '/data', robot_name=robot_name)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Meronomy data via ROS2 topics for Maplab.")
    parser.add_argument('--dataset_num', type=str, default=None, help="Dataset version number (e.g., V1.6).")
    parser.add_argument('--robot_name', type=str, default=None, help="Name of the robot (e.g., Husky1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(dataset_num=args.dataset_num, robot_name=args.robot_name)
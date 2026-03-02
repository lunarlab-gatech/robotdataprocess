import argparse
from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData, CameraData, TransformationData
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str):

    # Extract RGB and Depth Data
    input_path = Path(input_dir).absolute() 
    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', 'camera_color_optical_frame')
    depth_data = ImageDataOnDisk.from_npy_files(input_path / robot_name / 'depth', 'camera_color_optical_frame')

    # Extract Odometry (W->L in NED) and convert to W->B in FLU
    odom_data = OdometryData.from_csv(input_path.parent / 'results' / 'LIO-SAM' / robot_name / 'odometry.csv', "world", "body", CoordinateFrame.NED, True, None)  
    H_B_to_L = TransformationData.from_HERCULES_settings_json(str(input_path / 'settings.json'), robot_name, "sensor", "LidarSensor1")
    odom_data.apply_transformation_right_side(H_B_to_L.invert().as_matrix())
    odom_data.to_coordinate_frame(CoordinateFrame.FLU)

    # Define Camera parameters
    left_camera_data = CameraData.from_user_mono('/cam0', 752, 480, 376, 376, 376, 240, CameraData.DistortionModel.RADIAL_TANGENTIAL)

    # Multiply all timestamps by 20 to simulate traveling 20 times slower
    for data in [left_image_data, depth_data, left_camera_data, left_camera_data, odom_data]:
        data.timestamps = data.timestamps * Decimal('20.0')
    odom_data.interpolate_to_hz(4)

    # Publish the data via ROS topics
    publish_data_ROS_multiprocess([left_image_data, depth_data, left_camera_data, left_camera_data, odom_data], 
                                  ['/'+robot_name+'/cam0/image_raw',
                                   '/'+robot_name+'/cam0/depth/image_raw',
                                   '/'+robot_name+'/cam0/camera_info',
                                   '/'+robot_name+'/cam0/depth/camera_info',
                                   '/tf'],
                                    [None, None, None, None, "TFMessage"], 
                                    [20, 20, 1, 1, 20],
                                    [2, 2, 1, 1, 1], ROSMsgLibType.ROSPY, True, verbose=True)

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
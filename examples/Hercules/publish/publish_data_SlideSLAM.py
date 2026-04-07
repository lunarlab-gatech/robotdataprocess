import argparse
from decimal import Decimal
import getpass
import numpy as np
from pathlib import Path
from robotdataprocess import ImageDataOnDisk, OdometryData, CoordinateFrame, ROSMsgLibType
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
from typing import Union

def publish_data(input_dir: str, dataset_num: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):

    # Convert to Path objects
    input_path = Path(input_dir).absolute()

    # Extract RGB and IMU from Hercules v1.5
    odom_data = OdometryData.from_csv(input_path.parent.parent / 'results' / 'LIO-SAM/' / robot_name / 'odometry.csv', 
                                        "world", "robot", CoordinateFrame.NED, True, None)
    pose_data = OdometryData.from_txt(input_path / 'pose_world_frame.txt', 
                                           'world', robot_name + '/ground_truth/odom', CoordinateFrame.NED, False)
    seg_data = ImageDataOnDisk.from_image_files(input_path / 'seg', '' + robot_name + '/cam0')
    depth_data = ImageDataOnDisk.from_npy_files(input_path / 'depth', '' + robot_name + '/cam0')

    # Get L->I transformation
    if dataset_num == "V2.3.C":
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
        raise NotImplementedError(f"H_L_to_I not defined for dataset_version {dataset_num}")
    
    # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)
    odom_data.apply_transformation_right_side(H_L_to_I_in_NED)

    # Convert frame from NED to FLU
    odom_data.to_FLU_frame()
    pose_data.to_FLU_frame()

    # Do linear interpolation to get 10 Hz odometry
    odom_data.interpolate_to_hz(10.0)

    # Crop the data
    if crop_data:
        odom_data.crop_data(Decimal('0.0'), end_time)
        pose_data.crop_data(Decimal('0.0'), end_time)
        seg_data.crop_data(Decimal('0.0'), end_time)
        depth_data.crop_data(Decimal('0.0'), end_time)

    # Publish the data via ROS topics 
    publish_data_ROS_multiprocess([odom_data, pose_data, seg_data, depth_data], 
                                  [f'/{robot_name}/odom', f'/{robot_name}/odom_gt/path', f'/{robot_name}/cam0/seg', f'/{robot_name}/cam0/depth'],
                                  [None, "Path", None, None],
                                  [10, 20, 20, 20],
                                  [1, 1, 1, 1],
                                   ROSMsgLibType.ROSPY, True, verbose=True)

def main(dataset_num: str, robot_name: str, crop_end_time: Union[float, None]):

    # Do bookkeeping for cropping
    if crop_end_time is not None:
        crop_data = True
        end_time = Decimal(crop_end_time)
    else:
        crop_data = False
        end_time = None

    # Publish data for the specified robot
    user = getpass.getuser()
    input_dir = '/home/' + user + '/data/Hercules_datasets/' + dataset_num + '/data/' + robot_name
    publish_data(input_dir, dataset_num, robot_name, crop_data, end_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Hercules data via ROS2 topics for Maplab.")
    parser.add_argument('--dataset_num', type=str, default=None, help="Dataset version number (e.g., V2.3.C).")
    parser.add_argument('--robot_name', type=str, default=None, help="Name of the robot (e.g., Husky1).")
    parser.add_argument('--crop_end_time', type=float, default=None, help="Optional end time (in seconds) to crop the data.")
    args = parser.parse_args()

    main(dataset_num=args.dataset_num, robot_name=args.robot_name, crop_end_time=args.crop_end_time)
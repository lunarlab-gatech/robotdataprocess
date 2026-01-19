#!/usr/bin/env python3
"""
Simple script to extract images and IMU data from a folder and export to a ROS1 bag.

Usage:
    python simple_extract_to_bag.py

Configure the paths in the main() function below.
"""

import getpass
from pathlib import Path
import shutil
from robotdataprocess import ImageDataInMemory, ImuData, LiDARData, CoordinateFrame
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper


def extract_to_bag(input_dir: str, output_bag: str, robot_name: str):
    """
    Extract images, IMU data, and LiDAR data from a folder and save to a ROS1 bag.

    Args:
        input_dir (str): Path to the input directory containing:
            - rgb/ folder with PNG/JPG images (filenames are timestamps)
            - synthetic_imu.txt file (format: timestamp lin_acc_x lin_acc_y lin_acc_z ang_vel_x ang_vel_y ang_vel_z)
            - lidar/ folder with .npy files (filenames are timestamps)
        output_bag (str): Path where the output ROS1 bag will be saved (should end in .bag)
        robot_name (str): Name of the robot (used for frame IDs)
    """

    # Convert to Path objects
    input_path = Path(input_dir).absolute()
    output_path = Path(output_bag).absolute()

    # Create temporary ROS2 bag path (ROS2 bags are directories)
    temp_ros2_bag = output_path.parent / (output_path.stem + "_temp_ros2")

    print(f"Loading data from: {input_path}")
    print(f"Output ROS1 bag will be saved to: {output_path}")

    # Load IMU data from text file
    print("\n1. Loading IMU data...")
    imu_data = ImuData.from_txt_file(
        input_path / 'synthetic_imu.txt',
        f'{robot_name}/imu',
        CoordinateFrame.NED
    )
    print(f"   Loaded {imu_data.len()} IMU measurements")

    # Load images from folder
    print("\n2. Loading left images...")
    left_image_data = ImageDataInMemory.from_image_files(
        input_path / 'rgb_stereo_left',
        f'camera_0.0'
    )
    print(f"   Loaded {left_image_data.len()} images")

    # Loading right images
    print("\n2.b Loading right images...")
    right_image_data = ImageDataInMemory.from_image_files(
        input_path / 'rgb_stereo_right',
        f'camera_1.0'
    )
    print(f"   Loaded {right_image_data.len()} images")

    # Load LiDAR data from .npy files
    print("\n2.c Loading LiDAR data...")
    lidar_data = LiDARData.from_npy_files(
        input_path / 'lidar',
        'lidar_sensor_0', #change this for more robots
        CoordinateFrame.FLU
    )
    print(f"   Loaded {lidar_data.len()} point clouds")

    print("\n3. Writing to temporary ROS2 bag...")
    # Write data to temporary ROS2 bag (required intermediate step)
    Ros2BagWrapper.write_data_to_rosbag(
        temp_ros2_bag,
        [imu_data, left_image_data, right_image_data, lidar_data],    # Data to write
        ['/imu0', '/cam0/image_raw', '/cam1/image_raw', '/lidar'],  # Topic names
        [None, None, None, None],                     # Use default message types
        None                              # No external message definitions
    )

    print("\n4. Converting ROS2 bag to ROS1 bag...")
    # Get the repository root path (where external_msgs_ros1 is located)
    repo_root = Path(__file__).parent.parent.parent
    external_msgs_ros1_path = repo_root / "external_msgs_ros1"

    # Create wrapper for the temp ROS2 bag and convert to ROS1
    bag_wrapper = Ros2BagWrapper(temp_ros2_bag, None)
    bag_wrapper.export_as_ros1(output_path, external_msgs_ros1_path)

    print("\n5. Cleaning up temporary ROS2 bag...")
    # Remove the temporary ROS2 bag directory
    if temp_ros2_bag.exists():
        shutil.rmtree(temp_ros2_bag)

    print(f"\n✓ Successfully created ROS1 bag at: {output_path}")
    print(f"  - IMU topic: /imu0 ({imu_data.len()} messages)")
    print(f"  - Image topic: /cam0/image_raw ({left_image_data.len()} messages)")
    print(f"  - Image topic: /cam1/image_raw ({right_image_data.len()} messages)")
    print(f"  - LiDAR topic: /lidar ({lidar_data.len()} messages)")

def main():
    # ========== CONFIGURE THESE PATHS ==========

    # Robot name (used for frame IDs in the bag)
    robot_name = 'Husky1'
    dataset_num = "V2.1.0"

    # Input directory containing your data
    user = getpass.getuser()
    # input_dir = '/media/' + user + '/T7/GT/SLAM/Hercules_datasets/' + dataset_num + '/data/' + robot_name

    # # Output bag path
    # output_bag = '/media/' + user + '/T7/GT/SLAM/Hercules_datasets/' + dataset_num + '/extract/bags_for_maplab/' + robot_name + '.bag'
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data/' + robot_name

    # Output bag path
    output_bag = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/extract/bags_for_maplab/' + robot_name + '.bag'

    # ==========================================

    # Run the extraction
    extract_to_bag(input_dir, output_bag, robot_name)


if __name__ == "__main__":
    main()

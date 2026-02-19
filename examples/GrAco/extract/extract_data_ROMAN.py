from decimal import Decimal
import getpass
import os
import shutil
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, ImageDataOnDisk, LiDARData, ImageData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def data_extraction(input_dir: str, robot_name: str,  skip_depth: bool = False, skip_rgb: bool = False):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_roman_baseline'
    
    # Extract LiDAR data
    if not skip_depth:
        lidar_data = LiDARData.from_ros2_bag(input_path / robot_name / robot_name, '/velodyne/points', CoordinateFrame.ENU)
        lidar_data.to_npy_files(output_path / robot_name / 'lidar')

    # Extract image data
    if not skip_rgb:
        # Save to image files
        temp_dir = output_path / robot_name / 'camera_left_temp'
        rgb_data = ImageDataInMemory.from_ros2_bag(input_path / robot_name / robot_name, 
                        '/camera_left/image_raw', temp_dir)
        rgb_data.to_image_files(output_path / robot_name / 'camera_left')

        # Delete the temporary .npy file
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main(): 
    # Enter desired configuration here
    dataset_num = "V1.0"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/GrAco_dataset/' + dataset_num + '/data'
    robot_names = ["aerial-07", "aerial-08"]

    # Run extraction for each robot
    for i in range(len(robot_names)):
        data_extraction(input_dir=input_dir, 
                        robot_name=robot_names[i],
                        skip_depth=False,
                        skip_rgb=False)
        
if __name__ == "__main__":
    main()
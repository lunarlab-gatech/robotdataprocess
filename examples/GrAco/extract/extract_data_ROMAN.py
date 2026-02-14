from decimal import Decimal
import getpass
import os
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, ImageDataOnDisk, LiDARData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def data_extraction(input_dir: str, robot_name: str,  skip_depth: bool = False, skip_rgb: bool = False):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_roman_baseline'
    
    # Extract LiDAR data
    if not skip_depth:
        lidar_data = LiDARData.from_ros2_bag(input_path / robot_name / robot_name, '/velodyne/points', CoordinateFrame.ENU)
        lidar_data.to_npy()

    # Extract image dat
    if not skip_rgb:
        rgb_data = ImageDataInMemory.from_ros2_bag(input_path / robot_name / robot_name, '/camera_left/image_raw')
        rgb_data.to_npy(output_path / robot_name / 'rgb')

    # Load the odometry data
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 
                                           robot_name + '/odom', robot_name + '/ground_truth/base_link', 
                                           CoordinateFrame.NED, False)

    # Convert to the FLU coordinate frame & crop
    pose_data.to_FLU_frame()

    # Save back to a csv file
    if os.path.exists(output_path / robot_name / 'poseGT.csv'):
                print("Deleting CSV file at this location previously...")
                os.remove(output_path / robot_name / 'poseGT.csv')
    os.makedirs(output_path / robot_name, exist_ok=True)
    pose_data.to_csv(output_path / robot_name / 'poseGT.csv', write_header=True)

def main(): 
    # Enter desired configuration here
    dataset_num = "V1.0"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data'
    robot_names = ["ground-01", "ground-06"]

    # Run extraction for each robot
    for i in range(len(robot_names)):
        data_extraction(input_dir=input_dir, 
                        robot_name=robot_names[i],
                        skip_depth=False,
                        skip_rgb=False)
        
if __name__ == "__main__":
    main()
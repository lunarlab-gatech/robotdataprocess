from decimal import Decimal
import getpass
from pathlib import Path
import shutil
from robotdataprocess import ImageDataInMemory, OdometryData, CoordinateFrame
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def extract_to_bag(input_dir: str, output_bag: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):

    # Convert to Path objects
    input_path = Path(input_dir).absolute()
    output_path = Path(output_bag).absolute()

    # Create temporary ROS2 bag path (ROS2 bags are directories)
    temp_ros2_bag = output_path.parent / (output_path.stem + "_temp_ros2")

    # Extract RGB and IMU from Hercules v1.5
    odom_data = OdometryData.from_csv(input_path.parent.parent / 'extract' / 'files_for_roman_baseline' 
                                      / robot_name / 'vins_result_no_loop_reformatted.csv', 
                                      'world', robot_name+"/odom", 
                                      CoordinateFrame.FLU, True)
    pose_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', 
                                           'world', robot_name + '/ground_truth/odom', CoordinateFrame.NED)
    seg_data = ImageDataInMemory.from_image_files(input_path / 'seg', '' + robot_name + '/cam0')
    depth_data = ImageDataInMemory.from_npy_files(input_path / 'depth', '' + robot_name + '/cam0')

    # Convert pose data to FLU frame
    pose_data.to_FLU_frame()

    # Crop the data
    if crop_data:
        odom_data.crop_data(Decimal('0.0'), end_time)
        pose_data.crop_data(Decimal('0.0'), end_time)
        seg_data.crop_data(Decimal('0.0'), end_time)
        depth_data.crop_data(Decimal('0.0'), end_time)

    # Write data to temporary ROS2 bag (required intermediate step)
    Ros2BagWrapper.write_data_to_ros2_bag(
        temp_ros2_bag,
        [odom_data, pose_data, seg_data, depth_data], 
        ['/odom', '/odom_gt/path', '/cam0/seg', '/cam0/depth'], 
        [None, "Path", None, None], 
        None)
    
    # Inform the user how to finish
    print("To finish, use the rosbags-convert command line tool to convert from a ROS2 bag to a ROS1 bag.")
    print("Ex: rosbags-convert --src <robot_name>_temp_ros2/ --dst <robot_name>.bag")

def main():
    robot_name = 'Drone2'
    dataset_num = "V2.1.0"
    crop_data = False
    end_time = None

    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data/' + robot_name
    output_bag = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/extract/bags_for_slideslam/' + robot_name + '.bag'

    extract_to_bag(input_dir, output_bag, robot_name, crop_data, end_time)

if __name__ == "__main__":
    main()
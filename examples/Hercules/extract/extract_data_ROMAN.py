from decimal import Decimal
import getpass
import os
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, ImageDataOnDisk
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def data_extraction(input_dir: str, robot_name: str, 
                    start_time: Decimal, end_time: Union[Decimal, None], 
                    skip_depth: bool = False, skip_rgb: bool = False):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_roman_baseline'
    
    # Extract depth data from Hercules V1.5 from individual .npy files to a single .npy file
    if not skip_depth:
        depth_data = ImageDataOnDisk.from_npy_files(input_path / robot_name / 'depth', 'front_center_DepthPerspective')
        depth_data.crop_data(start_time, end_time)
        depth_data.to_npy(output_path / robot_name / 'depth')

    # Extract image data from Hercules V1.5 to .npy
    if not skip_rgb:
        rgb_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', 'front_center_Scene')
        rgb_data.crop_data(start_time, end_time)
        rgb_data.to_npy(output_path / robot_name / 'rgb')

    # Load the odometry data
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 
                                           robot_name + '/odom', robot_name + '/ground_truth/base_link', 
                                           CoordinateFrame.NED, False)

    # Convert to the FLU coordinate frame & crop
    pose_data.to_FLU_frame()
    pose_data.crop_data(start_time, end_time)

    # Save back to a csv file
    if os.path.exists(output_path / robot_name / 'poseGT.csv'):
                print("Deleting CSV file at this location previously...")
                os.remove(output_path / robot_name / 'poseGT.csv')
    os.makedirs(output_path / robot_name, exist_ok=True)
    pose_data.to_csv(output_path / robot_name / 'poseGT.csv', write_header=True)

def main(): 
    # Enter desired configuration here
    dataset_nums = ["V2.4.F"]

    for dataset_num in dataset_nums:
        user = getpass.getuser()
        input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data'
        robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

        # Get crop times
        if dataset_num == "V2.4.C":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('382.85'), Decimal('390.90'), Decimal('1100.00'), Decimal('1190.35')]
        elif dataset_num == "V2.3.AP":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('772.15'), Decimal('741.45'), Decimal('1121.80'), Decimal('1193.80')]
        elif dataset_num == "V2.3.AC":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('1125.00'), Decimal('1118.80'), Decimal('1025.50'), Decimal('892.60')]
        elif dataset_num == "V2.3.C":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('4105.00'), Decimal('4105.00'), Decimal('4105.00'), Decimal('4105.00')]
        elif dataset_num == "V2.4.F":
            robot_crop_start_times = [Decimal('35.05'), Decimal('34.60'), Decimal('27.45'), Decimal('31.50')]
            robot_crop_end_times = [Decimal('575.55'), Decimal('762.35'), Decimal('898.10'), Decimal('906.85')]
        else:
            raise ValueError("Crop times not specified for this dataset number.")

        # Run extraction for each robot
        for i in range(len(robot_names)):
            data_extraction(input_dir=input_dir, 
                            robot_name=robot_names[i],
                            start_time=robot_crop_start_times[i],
                            end_time=robot_crop_end_times[i],
                            skip_depth=False,
                            skip_rgb=False)
        
if __name__ == "__main__":
    main()
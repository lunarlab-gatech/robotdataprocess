from decimal import Decimal
import getpass
import os
import shutil
from pathlib import Path
from robotdataprocess import ImageDataOnDisk, CoordinateFrame, LiDARData, CameraData, OdometryData, TransformationData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def format_flat_matrix(values, values_per_line: int) -> str:
    lines = []
    for i in range(0, len(values), values_per_line):
        chunk = values[i:i + values_per_line]
        lines.append(', '.join(f"{v:.15f}" for v in chunk))
    return "[" + ", \n      ".join(lines) + "]"

def data_extraction(input_dir: str, robot_name: str,  skip_depth: bool = False, skip_rgb: bool = False):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_roman_baseline'

    # Get Camera intrinsics
    robot_type = robot_name.split('-')[0]
    camera_data = CameraData.from_kalibr_mono(input_path / (robot_type + '-calibration') / 'stereo.yaml', 'cam0')

    # Get T_odombase_camera
    T_odombase_camera = TransformationData.from_GrAco_yaml(input_path / (robot_type + '-calibration') / 'stereo-imu.yaml', 'T_Imu_cam0')
    T_odombase_camera_str = format_flat_matrix(T_odombase_camera.as_matrix().flatten(), values_per_line=4)
    print(f"T_odombase_camera Matrix: \n {T_odombase_camera_str}")

    # Get T_camera_flu (also known as T_camera_odombase)
    T_camera_flu = T_odombase_camera.invert()
    T_camera_flu_str = format_flat_matrix(T_camera_flu.as_matrix().flatten(), values_per_line=4)
    print(f"T_camera_flu Matrix: \n {T_camera_flu_str}")

    # Get T_base_lidar
    T_base_lidar = TransformationData.from_GrAco_yaml(input_path / (robot_type + '-calibration') / 'imu-lidar.yaml', 'T_Imu_Lidar')
    T_base_lidar_str = format_flat_matrix(T_base_lidar.as_matrix().flatten(), values_per_line=4)
    print(f"T_base_lidar Matrix: \n {T_base_lidar_str}")

    # Extract LiDAR data
    if not skip_depth:
        lidar_data = LiDARData.from_ros2_bag(input_path / robot_name / robot_name, '/velodyne/points', CoordinateFrame.ENU)
        lidar_data.to_npy_files(output_path / robot_name / 'lidar')

    # Extract image data
    if not skip_rgb:
        # Load images
        temp_dir = output_path / robot_name / 'camera_left_temp'
        mono_data = ImageDataOnDisk.from_ros1_bag(input_path / robot_name / (robot_name + ".bag"),
                        '/camera_left/image_raw')
        
        # Crop to the LiDAR FOV
        mono_data.crop_images_to_LiDAR_FOV((-15, 15), camera_data)

        # Save in a npy file
        mono_data.to_npy(output_path / robot_name / 'camera_left')

        # Delete the temporary .npy file
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        # Print camera instrincis (after crop edits it)
        K_str = format_flat_matrix(camera_data.K.flatten(), values_per_line=3)
        D_str = format_flat_matrix(camera_data.D.flatten(), values_per_line=5)
        print(f"K Matrix: \n {K_str}")
        print(f"D Matrix: \n {D_str}")
        print(f"Image Height: {camera_data.height}")
        print(f"Image Width: {camera_data.width}")
    
    # Extract the odometry (using GT for now)
    odometry_data = OdometryData.from_txt(input_path / robot_name / (robot_name + ".txt"), 'world', 'Imu', CoordinateFrame.ENU, 
                                             False, [0, 1, 2, 3, 7, 4, 5, 6])
    if os.path.exists(output_path / robot_name / 'poseGT.csv'):
                print("Deleting CSV file at this location previously...")
                os.remove(output_path / robot_name / 'poseGT.csv')
    os.makedirs(output_path / robot_name, exist_ok=True)
    odometry_data.to_csv(output_path / robot_name / 'poseGT.csv', write_header=True)

def main(): 
    # Enter desired configuration here
    dataset_num = "V1.0"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/GrAco_dataset/' + dataset_num + '/data'
    robot_names = ["ground-01", "ground-06"]

    # Run extraction for each robot
    for i in range(len(robot_names)):
        data_extraction(input_dir=input_dir, 
                        robot_name=robot_names[i],
                        skip_depth=True,
                        skip_rgb=False)
        
if __name__ == "__main__":
    main()
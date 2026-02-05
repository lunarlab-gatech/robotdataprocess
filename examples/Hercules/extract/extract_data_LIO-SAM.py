from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def to_bag(input_dir: str, robot_name: str, start_time: Decimal, end_time: Decimal):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'bags_for_LIO-SAM'

    # Extract RGB and IMU from Hercules v1.5
    imu_data = ImuData.from_txt_file(input_path / robot_name / 'synthetic_imu_9axis_500Hz.txt', 'base_link', CoordinateFrame.NED, nine_axis=True)
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 'map', 'base_link', CoordinateFrame.NED, False)
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", "lidar_link", CoordinateFrame.NED)

    # Prepare LiDARData
    lidar_data.calculate_point_channels(16, -20, 20)
    lidar_data.make_dense()

    # Shift GT data to start at Identity to be roughly close to odometry output
    pose_data.shift_to_start_at_identity()

    # Crop the data
    imu_data.crop_data(start_time, end_time)
    pose_data.crop_data(start_time, end_time)
    lidar_data.crop_data(start_time, end_time)

    # Save it into a ROS2 bag
    Ros2BagWrapper.write_data_to_rosbag(output_path / robot_name,
                      [  imu_data,     lidar_data,     pose_data],
                      ['/imu_raw',  '/points_raw',    '/odom_gt'],
                      [      None,           None,        "Path"], 
                                                             None)
    
def main(): 
    # Enter desired configuration here
    dataset_num = "V2.4.C"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data'
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

    # LIO-SAM starts from zero as it desires static initialization
    if dataset_num == "V2.4.C":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('382.85'), Decimal('390.90'), Decimal('1100.00'), Decimal('1190.35')]
    elif dataset_num == "V2.3.AP":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('772.15'), Decimal('741.45'), Decimal('1121.80'), Decimal('1193.80')]
    elif dataset_num == "V2.3.AC":
        robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
        robot_crop_end_times = [Decimal('1125.00'), Decimal('1118.80'), Decimal('1025.50'), Decimal('892.60')]
    else:
        raise ValueError("Crop times not specified for this dataset number.")

    # Check validity of inputs
    assert len(robot_names) == len(robot_crop_end_times)
    num_robots = len(robot_names)

    # Run extraction for each robot
    for i in range(num_robots):
        to_bag(input_dir=input_dir,
               robot_name=robot_names[i],
               start_time=robot_crop_start_times[i],
               end_time=robot_crop_end_times[i])
        
if __name__ == "__main__":
    main()
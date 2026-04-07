from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper

def to_bag(input_dir: str, robot_name: str, start_time: Decimal, end_time: Decimal):

    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'bags_for_LIO-SAM'

    # Extract IMU, GT Pose, and LiDAR from HERCULES simulator
    imu_data = ImuData.from_txt(input_path / robot_name / 'synthetic_imu_9axis.txt', 'base_link', CoordinateFrame.NED, nine_axis=True)
    pose_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', 'map', 'base_link', CoordinateFrame.NED, False)
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", "lidar_link", CoordinateFrame.NED)

    # Prepare LiDARData for LIO-SAM (requires point channels and dense point cloud)
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
                      [      None,           None,        "Path"], None)
    
def main(): 
    # Enter desired configuration here
    dataset_num = "V1.2"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Meronomy_datasets/' + dataset_num + '/data'
    robot_names = ["Husky1", "Drone1"]
    robot_crop_start_times = [Decimal('0'), Decimal('0')]
    robot_crop_end_times = [None, None]

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
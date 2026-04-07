from decimal import Decimal
import getpass
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from typing import Union

def to_bag(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'bags_for_vins_mono'

    # Extract RGB and IMU from Hercules v1.5
    imu_data = ImuData.from_txt(input_path / robot_name / 'synthetic_imu.txt', '' + robot_name + '/base_link', CoordinateFrame.NED)
    pose_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', 'world', 'body', CoordinateFrame.NED)
    image_data = ImageDataInMemory.from_image_files(input_path / robot_name / 'rgb_stereo_left', '' + robot_name + '/front_center_Scene')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    pose_data.to_FLU_frame()
    pose_data.shift_to_start_at_identity()

    # Crop the data
    if crop_data:
        imu_data.crop_data(Decimal('0.0'), end_time)
        pose_data.crop_data(Decimal('0.0'), end_time)
        image_data.crop_data(Decimal('0.0'), end_time)

    # Save it into a ROS2 Humble bag
    Ros2BagWrapper.write_data_to_rosbag(output_path / robot_name,
             [imu_data, image_data,  pose_data,       pose_data],
             [  '/imu0',    '/cam0/image_raw', '/odom_gt', '/odom_gt/path'],
             [    None,       None, "Odometry",          "Path"], 
             None)

def main(): 
    # Enter desired configuration here
    dataset_num = "V2.1.2"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data'
    robot_names = ["Drone1"]
    robot_crop_end_times = [None] 

    # Check validity of inputs
    assert len(robot_names) == len(robot_crop_end_times)
    num_robots = len(robot_names)

    # Run extraction for each robot
    for i in range(num_robots):
        if robot_crop_end_times[i] == None: crop_data = False
        else: crop_data = True

        to_bag(input_dir=input_dir,
               robot_name=robot_names[i],
               crop_data=crop_data,
               end_time=robot_crop_end_times[i])
        
if __name__ == "__main__":
    main()
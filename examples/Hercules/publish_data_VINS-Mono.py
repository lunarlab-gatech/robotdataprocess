from decimal import Decimal
from pathlib import Path
from robotdataprocess import ImuData, OdometryData, CoordinateFrame
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk
from robotdataprocess.ros.Ros2Publisher import publish_data_ROS2_multiprocess
from typing import Union

def publish_data(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    # Check parameters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Extract RGB and IMU from Hercules
    input_path = Path(input_dir).absolute() 
    #imu_data = ImuData.from_txt_file(input_path / robot_name / 'synthetic_imu.txt', '' + robot_name + '/base_link', CoordinateFrame.NED)
    #pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 'world', 'body', CoordinateFrame.NED)
    image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb', '' + robot_name + '/front_center_Scene')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    # pose_data.to_FLU_frame()
    # pose_data.shift_to_start_at_identity()

    # Crop the data
    if crop_data:
        #imu_data.crop_data(Decimal('0.0'), end_time)
        #pose_data.crop_data(Decimal('0.0'), end_time)
        image_data.crop_data(Decimal('0.0'), end_time)

    # Publish the data via ROS2 topics
    publish_data_ROS2_multiprocess([image_data], ['/cam0/image_raw'])

def main(): 
    # Enter desired configuration here
    dataset_num = "V1.6"
    input_dir = '/media/dbutterfield3/T731/Hercules_datasets/' + dataset_num + '/data'
    robot_names = ["Drone1"]
    robot_crop_end_times = [None] 

    # Check validity of inputs
    assert len(robot_names) == len(robot_crop_end_times)
    num_robots = len(robot_names)

    # Run extraction for each robot
    for i in range(num_robots):
        if robot_crop_end_times[i] == None: crop_data = False
        else: crop_data = True

        publish_data(input_dir=input_dir,
                     robot_name=robot_names[i],
                     crop_data=crop_data,
                     end_time=robot_crop_end_times[i])
        
if __name__ == "__main__":
    main()
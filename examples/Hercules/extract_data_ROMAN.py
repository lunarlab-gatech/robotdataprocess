from decimal import Decimal
from pathlib import Path
from robotdataprocess import ImageData, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper

def data_extraction(input_dir: str, robot_name: str, crop_data: bool, end_time: Decimal | None, skip_depth: bool = False, skip_rgb: bool = False):
    # Check paramters
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")
    
    # Make directory paths
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_roman_baseline'
    
    # Extract depth data from Hercules V1.5 from individual .npy files to a single .npy file
    if not skip_depth:
        depth_data = ImageData.from_npy_files(input_path / robot_name / 'depth', 'front_center_DepthPerspective')
        if crop_data: 
            depth_data.crop_data(Decimal('0.0'), end_time)
        depth_data.to_npy(output_path / robot_name / 'depth')

    # Extract image data from Hercules V1.5 to .npy
    if not skip_rgb:
        rgb_data = ImageData.from_image_files(input_path / robot_name / 'rgb', 'front_center_Scene')
        if crop_data: 
            rgb_data.crop_data(Decimal('0.0'), end_time)
        rgb_data.to_npy(output_path / robot_name / 'rgb')

    # Load the odometry data
    pose_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', robot_name + '/odom', robot_name + '/ground_truth/base_link', CoordinateFrame.NED)

    # Convert to the FLU coordinate frame & crop
    pose_data.to_FLU_frame()
    if crop_data: 
        pose_data.crop_data(Decimal('0.0'), end_time)

    # Save back to a csv file
    pose_data.to_csv(output_path / robot_name / 'poseGT.csv')

def main(): 
    # Enter desired configuration here
    dataset_num = "V1.6"
    input_dir = '/home/dbutterfield3/Desktop/data/Hercules_datasets/' + dataset_num + '/data'
    robot_names = ["Drone1"]
    robot_crop_end_times = [None] 

    # Check validity of inputs
    assert len(robot_names) == len(robot_crop_end_times)
    num_robots = len(robot_names)

    # Run extraction for each robot
    for i in range(num_robots):
        if robot_crop_end_times[i] == None: crop_data = False
        else: crop_data = True

        data_extraction(input_dir=input_dir, 
                        robot_name=robot_names[i],
                        crop_data=crop_data,
                        end_time=robot_crop_end_times[i],
                        skip_depth=True,
                        skip_rgb=True)
        
if __name__ == "__main__":
    main()
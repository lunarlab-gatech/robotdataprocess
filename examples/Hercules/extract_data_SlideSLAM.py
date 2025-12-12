from decimal import Decimal
from pathlib import Path
import shutil
from robotdataprocess import ImageData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper


def extract_to_bag(input_dir: str, output_bag: str, robot_name: str, crop_data: bool, end_time: Decimal | None):

    # Convert to Path objects
    input_path = Path(input_dir).absolute()
    output_path = Path(output_bag).absolute()

    # Create temporary ROS2 bag path (ROS2 bags are directories)
    temp_ros2_bag = output_path.parent / (output_path.stem + "_temp_ros2")

    # Extract RGB and IMU from Hercules v1.5
    odom_data = OdometryData.from_txt_file(input_path / 'pose_world_frame.txt', 'world', 'body', CoordinateFrame.NED)
    seg_data = ImageData.from_image_files(input_path / 'seg', '' + robot_name + '/cam0')
    depth_data = ImageData.from_npy_files(input_path / 'depth', '' + robot_name + '/cam0')

    # Convert data from NED frame to FLU frame
    odom_data.to_FLU_frame()

    # Crop the data
    if crop_data:
        odom_data.crop_data(Decimal('0.0'), end_time)
        seg_data.crop_data(Decimal('0.0'), end_time)
        depth_data.crop_data(Decimal('0.0'), end_time)

    # Write data to temporary ROS2 bag (required intermediate step)
    Ros2BagWrapper.write_data_to_rosbag(
        temp_ros2_bag,
        [odom_data, seg_data, depth_data], 
        ['/odom', '/cam0/seg', '/cam0/depth'], 
        [None, None, None], 
        None)

    # Get the repository root path (where external_msgs_ros1 is located)
    repo_root = Path(__file__).parent.parent.parent
    external_msgs_ros1_path = repo_root / "external_msgs_ros1"

    # Create wrapper for the temp ROS2 bag and convert to ROS1
    bag_wrapper = Ros2BagWrapper(temp_ros2_bag, None)
    bag_wrapper.export_as_ros1(output_path, external_msgs_ros1_path)

    # Remove the temporary ROS2 bag directory
    if temp_ros2_bag.exists():
        shutil.rmtree(temp_ros2_bag)

    print(f"\n✓ Successfully created ROS1 bag at: {output_path}")

def main():
    robot_name = 'Drone1'
    dataset_num = "V1.6"
    crop_data = False
    end_time = None

    input_dir = '/media/dbutterfield3/T731/Hercules_datasets/' + dataset_num + '/data/' + robot_name
    output_bag = '/media/dbutterfield3/T731/Hercules_datasets/' + dataset_num + '/extract/bags_for_slideslam/' + robot_name + '.bag'

    extract_to_bag(input_dir, output_bag, robot_name, crop_data, end_time)

if __name__ == "__main__":
    main()
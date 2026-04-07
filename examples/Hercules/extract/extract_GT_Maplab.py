from decimal import Decimal
import getpass
import os
from pathlib import Path
from robotdataprocess import OdometryData, CoordinateFrame
from typing import Union


def data_extraction(input_dir: str, robot_name: str, crop_data: bool, end_time: Union[Decimal, None]):
    input_path = Path(input_dir).absolute()
    output_path = input_path.parent / 'extract' / 'files_for_maplab_baseline'
    pose_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', robot_name + '/odom', robot_name + '/ground_truth/base_link', CoordinateFrame.NED, False)

    # Convert to the FLU coordinate frame & crop
    pose_data.to_FLU_frame()
    if crop_data:
        assert end_time is not None, "end_time must be provided when crop_data is True"
        pose_data.crop_data(Decimal('0.0'), end_time)

    # Save back to a csv file
    if os.path.exists(output_path / robot_name / 'poseGT.csv'):
        print("Deleting CSV file at this location previously...")
        os.remove(output_path / robot_name / 'poseGT.csv')
    os.makedirs(output_path / robot_name, exist_ok=True)
    pose_data.to_csv(output_path / robot_name / 'poseGT.csv')


if __name__ == "__main__":
    dataset_num = "V2.1.0"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T7/GT/SLAM/Hercules_datasets/' + dataset_num + '/data'
    robot_name = "Drone1"

    data_extraction(
        input_dir=input_dir,
        robot_name=robot_name,
        crop_data=False,
        end_time=None,
    )

from decimal import Decimal
import getpass
import numpy as np
import os
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, TransformationData, ImageDataOnDisk
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():
    """ Reformat data from LIO-SAM to be used as input odometry for ROMAN """
    
    robot_names = ["ground-01", "ground-06"]
    dataset_version = "V1.0"

    # Do it for all files, robots, and datasets
    for robot_name in robot_names:
                
        # Load the odometry data from LIO-SAM
        user = getpass.getuser()
        robot_type: str = robot_name.split('-')[0]
        dataset_path: Path = Path('/media') / user / 'T73' / 'GrAco_dataset' / dataset_version
        npy_folder_path: Path = dataset_path / 'results' / 'Metric3D' / robot_type / robot_name / 'depth'
        depth_data = ImageDataOnDisk.from_npy_files(npy_folder_path, 'optical')

        # Save in a single npy file
        output_path: Path = dataset_path / 'extract' / 'files_for_roman_baseline' / robot_name / 'depth'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        depth_data.to_npy(output_path)

if __name__ == "__main__":
    main()
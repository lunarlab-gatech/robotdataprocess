from decimal import Decimal
import getpass
import numpy as np
import os
from pathlib import Path
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, TransformationData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():
    """ Reformat data from LIO-SAM to be used as input odometry for ROMAN """
    
    robot_names = ["ground-01", "ground-06"]
    dataset_version = "V1.0"
    file_name = 'odometryHighHertz.csv'

    # Do it for all files, robots, and datasets
    for robot_name in robot_names:
                
        # Load the odometry data from LIO-SAM
        user = getpass.getuser()
        robot_type: str = robot_name.split('-')[0]
        dataset_folder = '/media/' + user + '/T73/GrAco_dataset/' + dataset_version
        odom_data = OdometryData.from_csv(dataset_folder + '/results/LIO-SAM/' + robot_type + '/' + robot_name + '/' + file_name, 
                                        'world', 'lidar', CoordinateFrame.ENU, header_included=True, column_to_data=None)

        # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)'
        calib_name = robot_type + "-calibration"
        H_I_to_L = TransformationData.from_GrAco_yaml(str(Path(dataset_folder) / 'data' / calib_name / "imu-lidar.yaml"), "T_Imu_Lidar")
        H_L_to_I = H_I_to_L.invert()
        odom_data.apply_transformation_right_side(H_L_to_I.as_matrix())
        odom_data.child_frame_id = 'Imu'

        # Save the csv in a ROMAN friendly format
        output_path = '/media/' + user + '/T73/GrAco_dataset/'+dataset_version+'/extract/files_for_roman_baseline/' \
                        + robot_name + '/' + file_name.replace('.csv', '_LIO-SAM_ENU_frame.csv')
        if os.path.exists(output_path):
            print("Deleting CSV file at this location previously...")
            os.remove(output_path)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        odom_data.to_csv(output_path)

if __name__ == "__main__":
    main()
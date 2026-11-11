from decimal import Decimal
import getpass
import numpy as np
import os
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():
    robot_names = ["Drone1"]
    dataset_names = ["V2.3.C"]
    file_names = ['vins_result_loop.csv', 'vins_result_no_loop.csv']

    # Do it for all files, robots, and datasets
    for robot_name in robot_names:
        for dataset_name in dataset_names:
            for file_name in file_names:
                
                # Load the odometry data
                user = getpass.getuser()
                odom_data = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/'+dataset_name+'/results/vins_mono/'+robot_name+'/'+file_name, "odom", 'base_link', CoordinateFrame.FLU, False, None)

                # Since positions are in FLU but orientations are in NED rotated to FLU, lets fix that
                R_NED = np.array([[1,  0,  0],
                                [0, -1,  0],
                                [0,  0, -1]])
                R_NED_Q = R.from_matrix(R_NED)
                odom_data._ori_apply_rotation_left_side(R_NED_Q.inv())
                odom_data._ori_change_of_basis(R_NED_Q)

                # The timestamps are in ns for 'vins_result_loop.csv', so set to seconds
                if file_name == 'vins_result_loop.csv':
                    odom_data.timestamps = odom_data.timestamps / Decimal('1e9')

                # Save the csv in a ROMAN friendly format
                output_path = '/media/' + user + '/T73/Hercules_datasets/'+dataset_name+'/extract/files_for_roman_baseline/' \
                              + robot_name + '/' + file_name.replace('.csv', '_reformatted.csv')
                if os.path.exists(output_path):
                    print("Deleting CSV file at this location previously...")
                    os.remove(output_path)
                else:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                odom_data.to_csv(output_path)

if __name__ == "__main__":
    main()
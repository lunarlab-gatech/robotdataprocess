from decimal import Decimal
import getpass
import numpy as np
import os
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]
    dataset_version = "V2.4.C"
    file_name = dataset_version + '.txt'

    for robot_name in robot_names:
        print("\n=== Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version
        
        # Load the data
        input_coordinate_frame = CoordinateFrame.NED if "Husky" in robot_name else CoordinateFrame.FLU
        odom_data = OdometryData.from_tum(dataset_folder + '/results/FAST-LIVO2/' + robot_name + '/' + file_name, 
                                        "world", "robot", input_coordinate_frame)

        # Convert frame to FLU
        odom_data.to_coordinate_frame(CoordinateFrame.FLU)
        if "Drone" in robot_name: 
            # Drone are in FLU frame but the local coordinate frame is NED
            # Thus, this sets local coordinate frame to FLU as well.
            R_NED_TO_FLU = np.array([[1,  0,  0],
                                     [0, -1,  0],
                                     [0,  0, -1]])
            odom_data._ori_apply_rotation_right_side(R.from_matrix(R_NED_TO_FLU))

        # Save the csv in a ROMAN friendly format
        output_path = '/media/' + user + '/T73/Hercules_datasets/'+dataset_version+'/extract/files_for_roman_baseline/' \
                        + robot_name + '/' + 'odometry_FAST-LIVO2_FLU_frame.csv'
        if os.path.exists(output_path):
            print("Deleting CSV file at this location previously...")
            os.remove(output_path)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        odom_data.to_csv(output_path)

if __name__ == "__main__":
    main()
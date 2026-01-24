from decimal import Decimal
import getpass
import numpy as np
import os
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():
    """ Reformat data from LIO-SAM to be used as input odometry for ROMAN """
    
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]
    dataset_version = "V2.3.C"
    file_name = 'odometry.csv'

    # Do it for all files, robots, and datasets
    for robot_name in robot_names:
                
        # Load the odometry data from LIO-SAM
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version
        odom_data = OdometryData.from_csv(dataset_folder + '/results/LIO-SAM/' + robot_name + '/' + file_name, 
                                        "world", "robot", CoordinateFrame.NED, True, None)

        # Get L->I transformation
        if dataset_version == "V2.3.C":
            if "Husky" in robot_name:
                H_L_to_I_in_NED = np.array([[1.0,  0.0,  0.0,  0.0],
                                            [0.0,  1.0,  0.0,  0.0],
                                            [0.0,  0.0,  1.0, 0.85],
                                            [0.0,  0.0,  0.0,  1.0]])
            elif "Drone" in robot_name:
                H_L_to_I_in_NED = np.array([[1.0,  0.0,  0.0,  0.0],
                                            [0.0,  1.0,  0.0,  0.0],
                                            [0.0,  0.0,  1.0,  0.5],
                                            [0.0,  0.0,  0.0,  1.0]])
        else:
            raise NotImplementedError(f"H_L_to_I not defined for dataset_version {dataset_version}")
        
        # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)
        odom_data.apply_transformation_right_side(H_L_to_I_in_NED)

        # Convert frame from NED to FLU
        odom_data.to_FLU_frame()

        # Save the csv in a ROMAN friendly format
        output_path = '/media/' + user + '/T73/Hercules_datasets/'+dataset_version+'/extract/files_for_roman_baseline/' \
                        + robot_name + '/' + file_name.replace('.csv', '_LIO-SAM_FLU_frame.csv')
        if os.path.exists(output_path):
            print("Deleting CSV file at this location previously...")
            os.remove(output_path)
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        odom_data.to_csv(output_path)

if __name__ == "__main__":
    main()
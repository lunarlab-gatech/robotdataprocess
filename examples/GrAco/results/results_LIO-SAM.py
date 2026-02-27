from decimal import Decimal
import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["ground-01", "ground-06", "aerial-07", "aerial-08"]
    file_name = 'odometry.csv'

    for robot_name in robot_names:
        print("\n=== Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/GrAco_dataset/V1.0'
        
        # Load the data
        robot_type = robot_name[:-3]
        est_data = OdometryData.from_csv(dataset_folder + '/results/LIO-SAM/' + robot_type + '/' + robot_name + '/' + file_name, 
                                        "world", "lidar", CoordinateFrame.ENU, True, None)
        gt_data = OdometryData.from_txt_file(dataset_folder + "/data/" + robot_name + '/' + robot_name + '.txt', 
                                        "world", "imu", CoordinateFrame.ENU, False, [0, 1, 2, 3, 7, 4, 5, 6])
             
        # Get L->I transformation
        if "ground" in robot_name:
            H_L_to_I_in_ENU = np.array([[0.99991641, -0.01025064, -0.00787932, 0.0126906],
                                        [0.01018984,  0.9999183,  -0.00771833, 0.0207969],
                                        [0.00795779,  0.0076374,   0.99993917, -0.122356],
                                        [         0,          0,            0,         1]])
        elif "aerial" in robot_name:
            H_L_to_I_in_ENU = np.array([[ 0.99982005, -0.01884091, -0.00221193,  0.0246595 ],
                                        [ 0.018868  ,  0.99973825,  0.01293998,  0.00344856],
                                        [ 0.00196755, -0.01297939,  0.99991383, -0.180213  ],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        else:
            raise RuntimeError(f"H_L_to_I_in_ENU not defined for robot_name {robot_name}.")

        # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)
        est_data.apply_transformation_right_side(H_L_to_I_in_ENU)

        # Calculate RMS ATE, among other metrics
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1,   
                                                                            visualize=True, axes_interval=[5000, 50])
        print("\nMetrics for file: ", file_name)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'], "\n")

        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])


if __name__ == "__main__":
    main()
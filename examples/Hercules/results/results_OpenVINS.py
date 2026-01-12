import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]
    dataset_version = "V2.1.0"
    file_name = 'ov_estimate.txt'

    for robot_name in robot_names:
        # Load Estimated data from OpenVINS
        user = getpass.getuser()
        est_data = OdometryData.from_txt_file('/media/' + user + '/T73/Hercules_datasets/' + dataset_version 
                                              + "/results/openvins/" + robot_name+ '/' + file_name, "world", "robot", 
                                              CoordinateFrame.FLU, True, [0, 5, 6, 7, 4, 1, 2, 3])

        # Orientations are in NED rotated to FLU, lets fix that, but should be change of basis instead
        R_NED = np.array([[1,  0,  0],
                            [0, -1,  0],
                            [0,  0, -1]])
        R_NED_Q = R.from_matrix(R_NED)
        est_data._ori_apply_rotation(R_NED_Q.inv())
        est_data._ori_change_of_basis(R_NED_Q)

        # Load the GT Data
        gt_data = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_version + \
                                        "/extract/files_for_roman_baseline/" + robot_name +'/poseGT.csv', 
                                        "world", "robot", CoordinateFrame.FLU, True, None)

        # Calculate RMS ATE, among other metrics
        metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        print("\nMetrics for file: ", file_name)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("Robot: ", robot_name, "RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
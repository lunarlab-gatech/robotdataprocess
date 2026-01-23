import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["Drone1"]
    dataset_version = "V2.3.C"
    file_names = ['odometry.csv']

    for robot_name in robot_names:
        for file_name in file_names:
            print("\n=== Processing results for robot:", robot_name)
            user = getpass.getuser()
            dataset_folder = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version
                       
            est_data = OdometryData.from_csv(dataset_folder + '/results/LIO-SAM/' + robot_name + '/' + file_name, 
                                            "world", "robot", CoordinateFrame.NED, True, None)
            gt_data = OdometryData.from_csv(dataset_folder + "/extract/files_for_roman_baseline/" + robot_name + '/poseGT.csv', 
                                            "world", "robot", CoordinateFrame.FLU, True, None)
            
            # Convert frame to FLU
            est_data.to_FLU_frame()

            # Calculate RMS ATE, among other metrics
            metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1,   
                                                                                visualize=True, axes_interval=500)
            print("\nMetrics for file: ", file_name)
            print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
            print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'], "\n")

            print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
            print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])


if __name__ == "__main__":
    main()
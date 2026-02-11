import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    file_name = 'CameraTrajectory.txt'

    user = getpass.getuser()
    dataset_folder = '/home/dbutterfield3/Research/ros_workspaces/orb_slam3_ws/src/ORB_SLAM3/'
                
    est_data = OdometryData.from_txt_file(dataset_folder + file_name, "world", "robot", CoordinateFrame.NED, False, [0,1,2,3,5,6,7,4])
    gt_data = OdometryData.from_csv('/media/dbutterfield3/T73/EuRoC/data/ASL/vicon_room1/V1_01_easy/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv', "world", "robot", CoordinateFrame.NED, True, None)

    #est_data.visualize([gt_data], ["ORB_SLAM 3 Estimated Trajectory", "GT"], axes_interval=10000, axes_length=1)
    
    metrics_dictionary: dict = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1,   
                                                                        visualize=True, axes_interval=5000, axes_length=1)
    print("\nMetrics for file: ", file_name)
    print("Robot: ", "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("Robot: ", "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'], "\n")

    print("Robot: ", "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Robot: ", "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
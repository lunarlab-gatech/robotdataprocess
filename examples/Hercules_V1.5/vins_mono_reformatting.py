import numpy as np
import os
from robotdataprocess import ImageData, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():
    robot_name = "Drone2"

    # Load the odometry data
    odom_data = OdometryData.from_csv('/home/dbutterfield3/Research/ros_workspaces/vins_mono_ws/src/VINS-MONO-ROS2/output/hercules/'+robot_name+'/vins_result_no_loop.csv', "odom", 'base_link', CoordinateFrame.FLU, False, None)

    # Since positions are in FLU but orientations are in NED rotated to FLU, lets fix that
    R_NED = np.array([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]])
    R_NED_Q = R.from_matrix(R_NED)
    odom_data._ori_apply_rotation(R_NED_Q.inv())
    odom_data._ori_change_of_basis(R_NED_Q)

    # Save the csv in a ROMAN friendly format
    output_path = '/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/extract/files_for_roman_baseline/'+robot_name+'/vins_result_no_loop_reformatted.csv'
    if os.path.exists(output_path):
        print("Deleting CSV file at this location previously...")
        os.remove(output_path)
    odom_data.to_csv(output_path)

if __name__ == "__main__":
    main()
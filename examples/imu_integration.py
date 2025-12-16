from decimal import Decimal
import numpy as np
from pathlib import Path
from robotdataprocess.data_types.ImuData import ImuData, CoordinateFrame
from robotdataprocess.data_types.OdometryData import OdometryData

def main():
    # Enter desired configuration here
    dataset_num = "V2.0.1"
    input_dir = '/media/dbutterfield3/T74/Hercules_Datasets/Archive/' + dataset_num + '/data'
    robot_name = "Drone2"

    # Make directory paths
    input_path = Path(input_dir).absolute()

    # Extract IMU data and GT Pose data
    imu_data = ImuData.from_txt_file(input_path / robot_name / 'synthetic_imu.txt', robot_name + '/base_link', CoordinateFrame.NED)
    gt_odom_data = OdometryData.from_txt_file(input_path / robot_name / 'pose_world_frame.txt', 'world', robot_name + '/base_link', CoordinateFrame.NED)


    # Convert imu data to odometry via integration and visualize compared to GT
    initial_pos = np.array([5.0, 5.0, 0.912486], dtype=float)
    initial_vel = np.array([0.0, 0.0, 0.0], dtype=float)
    initial_ori = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    odom_data: OdometryData = imu_data.to_PathData(initial_pos, initial_vel, initial_ori, use_ang_vel=True).to_OdometryData('world', robot_name + '/base_link')
    odom_data.visualize([gt_odom_data], ["IMU Derived Odometry", "Ground Truth Odometry"], axes_interval=5000, axes_length=10.0)

if __name__ == "__main__":
    main()
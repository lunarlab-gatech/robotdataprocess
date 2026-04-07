import getpass
from pathlib import Path
from robotdataprocess.data_types.ImuData import ImuData, CoordinateFrame
from robotdataprocess.data_types.OdometryData import OdometryData

def main():
    # Enter desired configuration here
    dataset_num = "V2.3.C"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_Datasets/' + dataset_num + '/data'
    robot_name = "Husky2"

    # Make directory paths
    input_path = Path(input_dir).absolute()

    # Extract IMU data and GT Pose data
    imu_data = ImuData.from_txt(input_path / robot_name / 'synthetic_imu_9axis_500Hz.txt', robot_name + '/base_link', CoordinateFrame.NED, nine_axis=True)
    gt_odom_data = OdometryData.from_txt(input_path / robot_name / 'pose_world_frame.txt', 'world', robot_name + '/base_link', CoordinateFrame.NED, False)

    # Convert imu data to odometry via integration and visualize compared to GT
    initial_pos = gt_odom_data.positions[0]
    initial_vel = (gt_odom_data.positions[1] - gt_odom_data.positions[0]) / (gt_odom_data.timestamps[1] - gt_odom_data.timestamps[0])
    initial_ori = gt_odom_data.orientations[0]
    print("Initial Position: ", initial_pos)
    print("Initial Velocity: ", initial_vel)
    print("Initial Orientation (quat xyzw): ", initial_ori)

    odom_data: OdometryData = imu_data.to_PathData(initial_pos, initial_vel, None, use_ang_vel=False).to_OdometryData('world', robot_name + '/base_link')
    odom_data.visualize_3D([gt_odom_data], ["IMU Derived Odometry", "Ground Truth Odometry"], axes_interval=5000, axes_length=10.0)

if __name__ == "__main__":
    main()
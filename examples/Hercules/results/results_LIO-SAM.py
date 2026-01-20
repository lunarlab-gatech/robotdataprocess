import getpass
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame

def main():
    # Load the GT and estimated path data (for GRaCo dataset)
    robot_name = "ground-01"
    file_name = "odometry.csv"
    print("\n=== Processing results for robot:", robot_name)
    user = getpass.getuser()
    est_data = OdometryData.from_csv('/media/' + user + '/T73/GRaCo_dataset/results/LIO-SAM/ground/' 
                                     + robot_name + '/' + file_name,  "world", "robot", CoordinateFrame.FLU, True, None)
    gt_data = OdometryData.from_ros2_bag('/media/' + user + '/T73/GRaCo_dataset/data/ground/' 
                                     + robot_name + '/', '/gnss/ground_truth', CoordinateFrame.FLU)

    # Calculate RMS ATE, among other metrics
    metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True, axes_length=3)
    print("\nMetrics for file: ", file_name)
    print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

    print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
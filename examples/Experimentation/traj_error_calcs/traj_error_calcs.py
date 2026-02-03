import getpass
from robotdataprocess import OdometryData, PathData, CoordinateFrame


def main():
    robot1gt = OdometryData.from_csv("robot1.csv", "world", "robot", CoordinateFrame.FLU, False, None)
    robot1odom = OdometryData.from_csv("robot1_est.csv", "world", "robot", CoordinateFrame.FLU, False, None)
    # robot1gt.visualize([], ["R1 GT"], axes_interval=100, axes_length=5.0)
    # robot1odom.visualize([], ["R1 Odom"], axes_interval=100, axes_length=5.0)

    robot2gt = OdometryData.from_csv("robot2.csv", "world", "robot", CoordinateFrame.FLU, False, None)
    robot2odom = OdometryData.from_csv("robot2_est.csv", "world", "robot", CoordinateFrame.FLU, False, None)
    # robot2gt.visualize([], ["R2 GT"], axes_interval=100, axes_length=5.0)
    # robot2odom.visualize([], ["R2 Odom"], axes_interval=100, axes_length=5.0)

    print("ROBOT 1 METRICS:")
    metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(robot1gt, robot1odom, max_diff=0.1, visualize=True, axes_length=0.01)
    print("Combined RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("Combined RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Combined RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("Combined RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    print("ROBOT 2 METRICS:")
    metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(robot2gt, robot2odom, max_diff=0.1, visualize=True, axes_length=0.01)
    print("Combined RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("Combined RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Combined RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("Combined RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    print("COMBINED METRICS:")
    est_data_lst: list[OdometryData] = [robot1odom, robot2odom]
    gt_data_lst: list[OdometryData] = [robot1gt, robot2gt]

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data_combined: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data_combined: PathData = PathData.concatenate_PathData(gt_data_lst)

    #est_data_combined.visualize([gt_data_combined], ["Drone1 + Husky1 Maplab Results", "Ground Truth"], [10, 10], [40, 1000])

    metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data_combined, est_data_combined, max_diff=0.1, visualize=True, axes_length=0.01)
    print("Combined RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("Combined RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("Combined RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("Combined RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])


if __name__ == "__main__":
    main()
import getpass
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

def main():  
    dataset_names = ["V2.1.0"]
    robot_names = ["Husky1", "Husky2"] # NOTE: [0] must be hostRobot_ID==0, and [1] must be hostRobot_ID==1 (from SlideSLAM)
    robot_names_merged = robot_names[0] + "_" + robot_names[1]

    # Do it for all datasets and robots
    for dataset_name in dataset_names:
        # Since SlideSLAM is Decentralized, each robot has an estimated trajectory for all robots
        for robot_name in robot_names:
            print("=== Processing results as estimated by robot", robot_name, "===")

            # Get robot0 name and robot1 name
            robot0_name = robot_names[0]
            robot1_name = robot_names[1]
            
            # Load the odometry data
            user = getpass.getuser()
            est_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/results/slideslam/' + robot_names_merged + '/' + robot_name + '/trajectory_0.csv', "odom", 'base_link', CoordinateFrame.NED, True, [7, 0, 1, 2, 6, 3, 4, 5], reorder_data=True)
            est_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/results/slideslam/' + robot_names_merged + '/' + robot_name + '/trajectory_1.csv', "odom", 'base_link', CoordinateFrame.NED, True, [7, 0, 1, 2, 6, 3, 4, 5], reorder_data=True)
            est_data_lst: list[OdometryData] = [est_data_robot0, est_data_robot1]
            est_data_robot0.visualize([est_data_robot1], [robot0_name + " Results", robot1_name + " Results"], 10, 40)

            # Load the ground truth data
            gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot0_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
            gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot1_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
            gt_data_lst: list[OdometryData] = [gt_data_robot0, gt_data_robot1]

            # # Make the timestamps match and then concatenate
            est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
            est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
            gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

            # # Calculate RMS ATE, among other metrics
            metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
            print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
            print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

            print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
            print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
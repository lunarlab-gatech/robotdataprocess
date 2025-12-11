from decimal import Decimal
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from pprint import pprint

def main():
    # Load the GT and estimated path data
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

    for robot_name in robot_names:
        dataset_version = "V1.4.1"
        robot_folder = "/home/dbutterfield3/Desktop/data/Hercules_datasets/" + dataset_version + \
                    "/extract/files_for_roman_baseline/" + robot_name
        est_data = OdometryData.from_csv(robot_folder + '/vins_result_loop_reformatted.csv', 
                                        "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data = OdometryData.from_csv(robot_folder +'/poseGT.csv', 
                                        "world", "robot", CoordinateFrame.FLU, True, None)

        # Calculate RMS ATE, among other metrics
        metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("Robot: ", robot_name, "RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
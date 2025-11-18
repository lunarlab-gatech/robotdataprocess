from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from pprint import pprint

def main():
    # Load the GT and estimated path data
    robot_name = "Husky1"
    dataset_version = "V.1.4.1"
    robot_folder = "/home/dbutterfield3/Desktop/data/Hercules_datasets/" + dataset_version + \
                   "/extract/files_for_roman_baseline/" + robot_name
    gt_data = OdometryData.from_csv(robot_folder + '/vins_result_no_loop_reformatted.csv', 
                                    "world", "robot", CoordinateFrame.FLU, True, None)
    est_data = OdometryData.from_csv(robot_folder +'/poseGT.csv', 
                                     "world", "robot", CoordinateFrame.FLU, True, None)

    # Calculate RMS ATE, among other metrics
    metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1)
    pprint(metrics_dictionary)

if __name__ == "__main__":
    main()
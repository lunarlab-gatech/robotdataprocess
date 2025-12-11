from decimal import Decimal
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from pprint import pprint

def main():
    # Load the GT and estimated path data
    robot_names = ["Husky1", "Husky2"]

    for robot_name in robot_names:
        dataset_version = "V1.5"

        gt_csv = "/media/dbutterfield3/T731/Hercules_datasets/" + dataset_version + \
                    "/extract/files_for_roman_baseline/" + robot_name + '/poseGT.csv'
        rovioli_csv = "/media/dbutterfield3/T731/Hercules_datasets/" + dataset_version + \
                    "/results/maplab_results/rovioli/" + robot_name + '/estimated_poses.csv'
        
        est_data = OdometryData.from_csv(rovioli_csv, "world", "robot", CoordinateFrame.FLU, True, [0,1,2,3,7,4,5,6], separator=r'\s')
        gt_data = OdometryData.from_csv(gt_csv, "world", "robot", CoordinateFrame.FLU, True, None)

        # TODO: Update frame of ROVIOLI output from NED to FLU
        
        # Calculate RMS ATE, among other metrics
        metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True, axes_length=1, axes_interval=100)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])


if __name__ == "__main__":
    main()
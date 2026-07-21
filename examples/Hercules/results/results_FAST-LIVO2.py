from decimal import Decimal
import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]
    dataset_version = "V2.4.C"
    file_name = dataset_version + '.txt'

    for robot_name in robot_names:
        print("\n=== Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version
        
        # Load the data
        input_coordinate_frame = CoordinateFrame.NED if "Husky" in robot_name else CoordinateFrame.FLU
        est_data = OdometryData.from_tum(dataset_folder + '/results/FAST-LIVO2/' + robot_name + '/' + file_name, 
                                        "world", "robot", input_coordinate_frame)
        gt_data = OdometryData.from_csv(dataset_folder + "/extract/files_for_roman_baseline/" + robot_name + '/poseGT.csv', 
                                        "world", "robot", CoordinateFrame.FLU, True, None)

        # Crop the GT data to match the estimated data time range
        if dataset_version == "V2.4.C":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('382.85'), Decimal('390.90'), Decimal('1100.00'), Decimal('1190.35')]
        elif dataset_version == "V2.3.AP":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('772.15'), Decimal('741.45'), Decimal('1121.80'), Decimal('1193.80')]
        elif dataset_version == "V2.3.AC":
            robot_crop_start_times = [Decimal('0.0'), Decimal('0.0'), Decimal('0.0'), Decimal('0.0')]
            robot_crop_end_times = [Decimal('1125.00'), Decimal('1118.80'), Decimal('1025.50'), Decimal('892.60')]
        elif dataset_version == "V2.4.F":
            robot_crop_start_times = [Decimal('35.05'), Decimal('34.60'), Decimal('27.45'), Decimal('31.50')]
            robot_crop_end_times = [Decimal('575.55'), Decimal('762.35'), Decimal('898.10'), Decimal('906.85')]
        else:
            raise ValueError("Crop times not specified for this dataset number.")
        
        gt_data.crop_data(robot_crop_start_times[robot_names.index(robot_name)], 
                          robot_crop_end_times[robot_names.index(robot_name)])
            
        # Convert frame to FLU
        est_data.to_coordinate_frame(CoordinateFrame.FLU)
        if "Drone" in robot_name: 
            # Drone are in FLU frame but the local coordinate frame is NED
            # Thus, this sets local coordinate frame to FLU as well.
            R_NED_TO_FLU = np.array([[1,  0,  0],
                                     [0, -1,  0],
                                     [0,  0, -1]])
            est_data._ori_apply_rotation_right_side(R.from_matrix(R_NED_TO_FLU))

        # Calculate RMS ATE, among other metrics
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1,   
                                                                            visualize=False, axes_interval=2000)
        print("\nMetrics for file: ", file_name)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'], "\n")

        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
        print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])


if __name__ == "__main__":
    main()
from decimal import Decimal
import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from robotdataprocess import TransformationData
from scipy.spatial.transform import Rotation as R

def main():
    # Load the GT and estimated path data
    robot_names = ["Drone1"]
    dataset_version = "V1.1"
    file_name = 'odometryHighHertz.csv'

    for robot_name in robot_names:
        print("\n=== Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/Meronomy_datasets/' + dataset_version
        
        # Load Estimated Data
        est_data = OdometryData.from_csv(dataset_folder + '/results/LIO-SAM/' + robot_name + '/' + file_name, "world", "robot", CoordinateFrame.NED, True, None)
                        
        # LIO-SAM output is W->L. However, our GT is W->I. Thus, we need to convert it (W->I = W->L @ L->I)
        H_I_to_L = TransformationData.from_HERCULES_settings_json(dataset_folder + '/data/settings.json', robot_name, "sensor", "LidarSensor1")
        est_data.apply_transformation_right_side(H_I_to_L.invert().as_matrix())
        est_data.to_coordinate_frame(CoordinateFrame.FLU)

        # Load GT Data
        gt_data = OdometryData.from_txt(dataset_folder + "/data/" + robot_name + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
        gt_data.to_coordinate_frame(CoordinateFrame.FLU)

        # Calculate RMS ATE, among other metrics
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1,   
                                                                            visualize=True, axes_interval=10)
        print("\nMetrics for file: ", file_name)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse, "\n")

        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)
        print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)


if __name__ == "__main__":
    main()
from decimal import Decimal
import getpass
import numpy as np
import os
from robotdataprocess import ImageDataInMemory, ImuData, OdometryData, CoordinateFrame, PathData
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from scipy.spatial.transform import Rotation as R

def main():  
    dataset_names = ["V1.5"]

    # Do it for all files, robots, and datasets
    for dataset_name in dataset_names:
            
        # Load the odometry data
        user = getpass.getuser()
        est_data_husky1 = OdometryData.from_csv('/media/' + user + '/T731/Hercules_datasets/' + dataset_name + '/results/maplab_results/maplab/merged_map/vertex_poses_velocities_biases.csv', "odom", 'base_link', CoordinateFrame.NED, True, [0,3,4,5,6,7,8,9], filter=(' mission-id', ' 9d9572d8194a7f180900000000000000'))
        est_data_husky2 = OdometryData.from_csv('/media/' + user + '/T731/Hercules_datasets/' + dataset_name + '/results/maplab_results/maplab/merged_map/vertex_poses_velocities_biases.csv', "odom", 'base_link', CoordinateFrame.NED, True, [0,3,4,5,6,7,8,9], filter=(' mission-id', ' 38a88adc194a7f180900000000000000'))
        est_data_lst: list[OdometryData] = [est_data_husky1, est_data_husky2]
        for est_data in est_data_lst:
            est_data.timestamps = est_data.timestamps / Decimal('1e9')  # Convert from ns to s
        est_data_husky1.visualize([est_data_husky2], ["Husky1 Maplab Results","Husky2 Maplab Results"], 10, 40)

        # Load the ground truth data
        gt_data_husky1 = OdometryData.from_csv('/media/' + user + '/T731/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/Husky1/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_husky2 = OdometryData.from_csv('/media/' + user + '/T731/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/Husky2/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_lst: list[OdometryData] = [gt_data_husky1, gt_data_husky2]

        # Make the timestamps match and then concatenate
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
        est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # Since positions are in FLU but orientations are in NED rotated to FLU, lets fix that
        R_NED = np.array([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]])
        R_NED_Q = R.from_matrix(R_NED)
        est_data._ori_apply_rotation(R_NED_Q.inv())
        est_data._ori_change_of_basis(R_NED_Q)
        est_data.frame = CoordinateFrame.FLU    
        est_data.visualize([gt_data], ["Husky1+Husky2 Maplab Results", "Ground Truth"], [10, 10], [40, 1000])

        # Calculate RMS ATE, among other metrics
        metrics_dictionary: dict = OdometryData.calculate_trajectory_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
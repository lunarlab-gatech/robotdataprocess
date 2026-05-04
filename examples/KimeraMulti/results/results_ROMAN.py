import getpass
from pathlib import Path
import re
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

def main():  

    # Set experiment configuration
    user = getpass.getuser()
    dataset_folder = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset'
    robot_names_text = ["acl_jackal","acl_jackal2","sparkal1","sparkal2","hathor","thoth", "apis","sobek"]
    dataset_number = "1208"
    repository_with_results = "roman"
    sequence_name = "Medium/Hybrid/acl_jackal_acl_jackal2_sparkal1_sparkal2_hathor_thoth_apis_sobek"

    # Load the estimated data
    est_data_dir = Path('/home/') / user / 'Research' / repository_with_results / 'kimera_multi_output' / sequence_name
    est_data_lst: list[OdometryData] = []
    for rn in robot_names_text:
        est_data = OdometryData.from_csv(est_data_dir / 'offline_rpgo' / (rn + ".csv"), "map", 'robot', 
                                CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        est_data_lst.append(est_data)

    # Load the ground truth data
    gt_data_lst: list[OdometryData] = []
    for rn in robot_names_text:
        gt_data = OdometryData.from_csv(dataset_folder / 'data' / 'ground_truth' / dataset_number / (rn + '_gt_odom.csv'), 
                                        "world", "robot", CoordinateFrame.FLU, True, None, ts_in_ns=True)
        gt_data_lst.append(gt_data)

    # Calculate individual RMS ATE, among other metrics
    for i in range(len(est_data_lst)):
        print("=========== Individual Trajectory", robot_names_text[i], "for dataset: ============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_lst[i], est_data_lst[i], 
                                                                                max_diff=0.1, visualize=False)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    # Calculate merged RMS ATE:
    if len(est_data_lst) > 1:
        # Make the timestamps match and then concatenate
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
        est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # Calculate RMS ATE, among other metrics
        print("\n========== Merged Trajectories for dataset: ==========")
        metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, 
                                                                        est_data, max_diff=0.1, visualize=True)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

if __name__ == "__main__":
    main()
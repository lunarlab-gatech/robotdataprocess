import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame

def main():
    # Set dataset configuration
    user = getpass.getuser()
    dataset_names = ["V2.4.C"] * 6
    run_names = ["peachy-sweep-1", "snowy-sweep-2", "winter-sweep-3", "light-sweep-4", "lively-sweep-5","solar-sweep-6"]

    # Calculate lc errors for each run and robot pair
    all_error_dicts = {}
    all_inlier_masks = {}
    for dataset_name, run_name in zip(dataset_names, run_names):

        # Get robot name pair for this configuration
        run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '/' + run_name)
        align_dir = run_folder / 'align'
        for subdir in align_dir.iterdir():
            if not subdir.is_dir(): continue
            parts = subdir.name.split('_')
            if len(parts) != 2: continue
            if parts[0] != parts[1]:
                robot_names_pair = (parts[0], parts[1])
                break

        # Load Loop Closure data
        lc_data = LoopClosureData.from_json(run_folder / 'align' / (robot_names_pair[0] + '_' + robot_names_pair[1]) / 'align.json')
        lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' /'inlier_lc_inter_robot.g2o', run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt', names_override=(robot_names_pair[0], robot_names_pair[1]))
        
        # Merge the datas so we know which are inliers vs. outliers
        lc_data.label_inliers_via_other_LoopClosureData(lc_data_inlier)
        lc_data.round_timestamps(2)

        # Load the GT data for both robots
        gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names_pair[0] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names_pair[1] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_dict: dict[str, OdometryData] = {robot_names_pair[0]: gt_data_robot0, robot_names_pair[1]: gt_data_robot1}
        
        # Calculate the errors for the loop closures
        all_error_dicts[run_name] = lc_data.calculate_errors(gt_data_dict)
        all_inlier_masks[run_name] = lc_data.detected_inliers

    # Visualize the results
    labels = list(all_error_dicts.keys())
    errors_list = list(all_error_dicts.values())
    inliers_list = list(all_inlier_masks.values())
    LoopClosureData.visualize_error_scatter(errors_list, labels, inliers_list, max_rotation_frac=1.0, max_translation_frac=1.0)

if __name__ == "__main__":
    main()
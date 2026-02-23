import getpass
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame

def main():
    # Set dataset configuration
    user = getpass.getuser()
    run_names = ["In_paper", "run-0e044ed8"]
    label_names = ["ROMAN", "MeronomyGraph"]
    dataset_names = ["V1.0"] * len(run_names)

    # Calculate lc errors for each run and robot pair
    all_error_dicts = {}
    all_inlier_masks = {}
    for dataset_name, run_name in zip(dataset_names, run_names):

        # Get robot name pair for this configuration
        run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/GrAco_' + dataset_name + '/' + run_name)
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

        # Round timestamps to allow proper matching
        lc_data.round_timestamps(4)
        lc_data_inlier.round_timestamps(4)

        # Merge the datas so we know which are inliers vs. outliers
        lc_data.label_inliers_via_other_LoopClosureData(lc_data_inlier)


        # Load the GT data for both robots
        gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                robot_names_pair[0] + '/' + robot_names_pair[0] + '.txt', "world", "robot", CoordinateFrame.ENU, 
                                False, [0, 1, 2, 3, 7, 4, 5, 6])
        gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/' + \
                                robot_names_pair[1] + '/' + robot_names_pair[1] + '.txt', "world", "robot", CoordinateFrame.ENU,  
                                False, [0, 1, 2, 3, 7, 4, 5, 6])
        gt_data_dict: dict[str, OdometryData] = {robot_names_pair[0]: gt_data_robot0, robot_names_pair[1]: gt_data_robot1}
        
        # Calculate the errors for the loop closures
        all_error_dicts[run_name] = lc_data.calculate_errors(gt_data_dict)
        all_inlier_masks[run_name] = lc_data.detected_inliers

    # Visualize the results
    errors_list = list(all_error_dicts.values())
    inliers_list = list(all_inlier_masks.values())
    LoopClosureData.visualize_error_scatter(errors_list, label_names, inliers_list, max_rotation_frac=1.0, max_translation_frac=1.0, trans_err_in_target=3.0, show_plots=False, rot_err_in_target=6.0, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')

if __name__ == "__main__":
    main()
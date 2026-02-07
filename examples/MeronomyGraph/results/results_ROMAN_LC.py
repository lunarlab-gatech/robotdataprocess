import getpass
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame

def main():
    # Set dataset configuration
    user = getpass.getuser()
    dataset_name = "V2.3.AC"
    run_names = ["comic-sweep-1", "lucky-sweep-3", "magic-sweep-4", "morning-sweep-5"]
    robot_names = [["Husky1", "Husky2"],
                   ["Husky1", "Drone2"],
                   ["Husky2", "Drone1"],
                   ["Husky2", "Drone2"]]

    # Calculate lc errors for each run and robot pair
    all_error_dicts = {}
    for run_name, robot_names_pair in zip(run_names, robot_names):
        lc_data = LoopClosureData.from_json('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' \
                                + dataset_name + '/' + run_name + '/align/' + robot_names_pair[0] + '_' + robot_names_pair[1] + '/align.json')
        
        # Load the GT data for both robots
        gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names_pair[0] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot_names_pair[1] + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_dict: dict[str, OdometryData] = {robot_names_pair[0]: gt_data_robot0, robot_names_pair[1]: gt_data_robot1}
        
        # Calculate the errors for the loop closures
        error_dict = lc_data.calculate_errors(gt_data_dict)
        all_error_dicts[run_name] = error_dict

    # Visualize the results
    labels = list(all_error_dicts.keys())
    errors_list = list(all_error_dicts.values())
    #LoopClosureData.visualize_errors(errors_list, labels, bins=15)
    #LoopClosureData.visualize_success_rate(errors_list, labels, num_thresholds=1000, max_rotation_frac=1.0, max_translation_frac=1.0)
    LoopClosureData.visualize_error_scatter(errors_list, labels, max_rotation_frac=0.25, max_translation_frac=0.25)

if __name__ == "__main__":
    main()
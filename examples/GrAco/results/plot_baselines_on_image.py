import getpass
from matplotlib.colors import to_hex
from robotdataprocess import OdometryData, PathData
from utils import LoadDataResult, print_errors
from results_GAC_Mapping import load_data_GAC_Mapping
from results_ROMAN import load_data_ROMAN
import seaborn as sns

def main():  
    dataset_name = "V1.0"
    robot_names = ["aerial-07", "aerial-08"]

    # ============================= GAC-Mapping =============================
    experiment_name = robot_names[0][0].capitalize() + robot_names[0][-2:] + '-' + robot_names[1][0].capitalize() + robot_names[1][-2:]
    data: LoadDataResult = load_data_GAC_Mapping(dataset_name, experiment_name, robot_names[0], robot_names[1])

    # Make the timestamps match and then concatenate
    data.est_data_lst, data.gt_data_lst = PathData.make_start_and_end_times_match(data.est_data_lst, data.gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(data.est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(data.gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== GAC-Mapping Merged Trajectories for dataset: ", dataset_name, "==========")
    metrics_dictionary, est_data_align_gac, gt_data_align_gac = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    print_errors(metrics_dictionary)

    # ============================= ROMAN =============================
    run_name = 'In_paper'
    data: LoadDataResult = load_data_ROMAN(dataset_name, run_name, robot_names[0], robot_names[1])

    # Make the timestamps match and then concatenate
    data.est_data_lst, data.gt_data_lst = PathData.make_start_and_end_times_match(data.est_data_lst, data.gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(data.est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(data.gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== ROMAN Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
    metrics_dictionary, est_data_align_roman, gt_data_align_roman = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    print_errors(metrics_dictionary)

    # ============================= Visualize both on the same map =============================
    # Get environment image path
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/environment.png'
    x_edge = 691.216296 

    # Plot the results in 2D
    dataList =  [gt_data_align_roman, est_data_align_roman, est_data_align_gac]
    isGTList =  [True, False, False]
    nameList =  ["GT", "ROMAN", "GAC-Mapping"]
    palette = sns.color_palette("bright", len(dataList) - 1)
    colorList =  ["#000000"] + [to_hex(c) for c in palette]
    PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=3.0, show_grid=False, 
                       background_image_path=image_path, background_image_x_edge=x_edge, gt_color_lightness_range_val=15,
                       background_image_extent_offsets=(55, 80), no_border=True,
                       save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')


if __name__ == "__main__":
    main()
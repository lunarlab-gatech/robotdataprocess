import getpass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

def main():  
    # Set all data to use
    dataset_names = ["V2.4.C", "V2.3.AP", "V2.3.AC", "V2.4.F"]
    run_names = ['peachy-sweep-1', 'woven-sweep-5', 'brisk-sweep-2', 'restful-sweep-6']
    robot_names_list = [["Husky1", "Husky2"],
                        ["Husky2", "Drone2"],
                        ["Husky1", "Drone1"],
                        ["Drone1", "Drone2"]]

    # Create the 2x2 plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    # Define colors
    colors = ["#5FF598", "#F5756E", "#F5DA62", "#6262F5"] # Original
    colors = ['#00FF90', '#FF006F', '#EFFF00', '#1000FF'] # Neon
    colors = ['#1EE18E', '#E11E71', '#D3E11E', '#2C1EE1'] # Desaturated
    colors = ["#1EE15F", "#E11E28", "#F0F02A", "#1B0ED5"]

    # Generate the plot for each sequence
    for i, dataset_name, run_name, robot_names in zip(range(4), dataset_names, run_names, robot_names_list):

        # Get robot0 name and robot1 name
        robot0_name = robot_names[0]
        robot1_name = robot_names[1]
        
        # Load the estimated data
        user = getpass.getuser()
        est_data_robot0 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '/' + run_name+ '/offline_rpgo/' + robot0_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        est_data_robot1 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '/' + run_name+ '/offline_rpgo/' + robot1_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        est_data_lst: list[OdometryData] = [est_data_robot0, est_data_robot1]

        # Load the ground truth data
        gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot0_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot1_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_lst: list[OdometryData] = [gt_data_robot0, gt_data_robot1]

        # Make the timestamps match and then concatenate
        est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
        est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # Calculate RMS ATE, among other metrics
        print("\n========== Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
        metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
        print("RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        print("RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse)
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)

        # Seperate the aligned trajectories into their single-robot forms
        gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
        gt_data_align_robot0 = gt_data_align_list[0]
        gt_data_align_robot1 = gt_data_align_list[1]

        est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
        est_data_align_robot0 = est_data_align_list[0]
        est_data_align_robot1 = est_data_align_list[1]

        # Get environment image path
        image_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/environment.png'
        if dataset_name in "V2.3.AP":  x_edge = 350
        elif dataset_name in "V2.4.C": x_edge = 300
        elif dataset_name in "V2.3.AC": x_edge = 500
        elif dataset_name in "V2.4.F": x_edge = 150
        else:
            raise RuntimeError(f"x_edge not defined for {dataset_name}.")

        # Define the mapping from robot name to color and robot_name to new name
        name_map: dict = {
            "Husky1": "UGV1",
            "Husky2": "UGV2",
            "Drone1": "UAV1",
            "Drone2": "UAV2"
        }
        robot_name_to_color: dict = {
            "UGV1": colors[0],
            "UGV2": colors[1],
            "UAV1": colors[2],
            "UAV2": colors[3]
        }

        # Plot the results in 2D (Configuration for Figure 10)
        dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
        isGTList =  [                False,                 True,                  False,                  True]
        nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]

        loc = "bottom-right" if i !=0 else "top-right"
        if i == 0: light_val = 5
        elif i == 1: light_val = 5
        elif i == 2: light_val = 6
        else: light_val = 13
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.5, show_grid=False,  
                              disable_y_label=bool(i%2), disable_x_label=int(i/2) < 1, no_border=True, legend=False, 
                              google_maps_scale_bar=True, google_maps_scale_bar_loc=loc, gt_color_lightness_range_val=light_val,
                              background_image_path=image_path, background_image_x_edge=x_edge, ax=axs[i])
    
    # Robot color legend entries
    spacer = Patch(visible=False)
    legend_handles = [
        Patch(facecolor=colors[0], edgecolor="none", label="UGV1"),
        Patch(facecolor=colors[1], edgecolor="none", label="UGV2"),
        Patch(facecolor=colors[2], edgecolor="black", linewidth=0.8, label="UAV1"),
        Patch(facecolor=colors[3], edgecolor="none", label="UAV2"),
        Line2D([0], [0], color="gray", lw=3, linestyle=":", label="GT"),
        Line2D([0], [0], color="gray", lw=3, linestyle="-", label="Est."),
    ]

    # Add the legend below the figure
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=8,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.48, 1.05),
    )

    # Save the resulting figure
    fig.tight_layout()
    fig.savefig("combined.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
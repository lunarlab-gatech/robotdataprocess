import getpass
import itertools
from multiprocessing import Pool
import re
import sys
from pathlib import Path
import pandas as pd
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from utils.visualization import save_styled_tables
from results_ROMAN_LC import calculate_LC_errors_ROMAN

def _pair_label(name_a: str, name_b: str) -> str:
    def abbrev(n):
        m = re.match(r'([A-Za-z]+)(\d+)', n)
        return (m.group(1)[0].upper() + m.group(2)) if m else n
    return abbrev(name_a) + abbrev(name_b)

def calculate_merged_ate(dataset_name: str, method: str, robot_names: list, visualize: bool = False, do_individual_calcs: bool = False) -> float:
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]
    run_name = "_".join(robot_names)

    # Load the estimated data
    user = getpass.getuser()
    est_data_robot0 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method + '/' + run_name+ '/offline_rpgo/' + robot0_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_robot1 = OdometryData.from_csv('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method + '/' + run_name+ '/offline_rpgo/' + robot1_name + '.csv', "map", 'robot0', CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
    est_data_lst: list[OdometryData] = [est_data_robot0, est_data_robot1]

    # Load the ground truth data
    gt_data_robot0 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot0_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
    gt_data_robot1 = OdometryData.from_csv('/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/extract/files_for_roman_baseline/' + robot1_name + '/poseGT.csv', "world", "robot", CoordinateFrame.FLU, True, None)
    gt_data_lst: list[OdometryData] = [gt_data_robot0, gt_data_robot1]

    # Calculate individual RMS ATE
    if do_individual_calcs:
        # TODO: Need to make start and end times match before individual RMS ATE as well;
        # if we ever use those results in a paper.

        print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, run_name, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

        print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, run_name, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False)
        print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
        print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, run_name, "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=visualize)
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'], "\n")

    if visualize:
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
            "UGV1": "#1EE15F",
            "UGV2": "#E11E28",
            "UAV1": "#F0F02A",
            "UAV2": "#1B0ED5",
        }

        # Plot the results in 2D (Configuration for Figure 10)
        dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
        isGTList =  [                False,                 True,                  False,                  True]
        nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True,
                           background_image_path=image_path, background_image_x_edge=x_edge,
                           save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

        # Configuration for Figure 2
        # PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=4.0, show_grid=False,
        #                     background_image_path=image_path, background_image_x_edge=x_edge, legend=False, no_border=True,
        #                     save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

    return metrics_dictionary['APE']['translation_part']['rmse']


def main():
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    robot_pairs = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN_NM", "MG_SS_3", "MG_SS_3_POA"]
    dataset_name = "V2.4.C"

    tasks = [(dataset_name, run_name, list(pair))
             for pair in robot_pairs
             for run_name in run_names]

    with Pool() as pool:
        results = pool.starmap(calculate_merged_ate, tasks)

    table_data: dict[str, dict[str, float]] = {run: {} for run in run_names}
    for (_, run_name, pair), ate in zip(tasks, results):
        col = _pair_label(*pair)
        table_data[run_name][col] = ate

    cols = [_pair_label(*p) for p in robot_pairs]

    inlier_lc_total: dict[str, dict[str, int]] = {run: {} for run in run_names}
    for pair in robot_pairs:
        col = _pair_label(*pair)
        for run_name in run_names:
            _, inlier_errs = calculate_LC_errors_ROMAN(dataset_name, run_name, list(pair), only_inter_lc=True)
            inlier_lc_total[run_name][col] = len(inlier_errs['translation_errors'])

    RUN_DISPLAY_NAMES = {
        "ROMAN_NM":   "NM + ROMAN",
        "MG_TS_2-4":  "NM + MG (Two Stage - 2/4 req)",
        "MG_TS_3-4":  "NM + MG (Two Stage - 3/4 req)",
        "MG_SS_3":    "NM + MG (Single Stage - 3 req)",
        "MG_SS_3_POA":"NM + MG (SS3) + POA"
    }
    def make_df():
        rows = {}
        for run in run_names:
            row = {}
            for col in cols:
                if inlier_lc_total[run].get(col, -1) == 0:
                    row[col] = "---"
                else:
                    row[col] = f"{table_data[run][col]:.2f}"
            rows[RUN_DISPLAY_NAMES.get(run, run)] = row
        return pd.DataFrame(rows).T

    def make_rank_df():
        # Negate so that lower ATE → higher rank value → sorted first by _col_rank_groups
        return pd.DataFrame(
            {RUN_DISPLAY_NAMES.get(run, run): {col: -table_data[run][col] for col in cols}
             for run in run_names}
        ).T

    dfs = [
        ("Merged RMS ATE (m)", make_df(), [make_rank_df()]),
    ]

    save_styled_tables(
        dfs,
        '/home/dbutterfield3/Research/robotdataprocess/ate_table.pdf',
        cell_is_red=lambda s: s == "---" or float(s) > 20)


if __name__ == "__main__":
    main()

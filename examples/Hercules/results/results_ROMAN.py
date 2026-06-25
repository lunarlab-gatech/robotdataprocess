import getpass
import itertools
from multiprocessing import Pool
import sys
from pathlib import Path
import pandas as pd
from robotdataprocess import LoopClosureData, OdometryData, PathData

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from utils.visualization import save_styled_tables
from results_ROMAN_LC import calculate_LC_errors_ROMAN, load_LC_data_ROMAN
from utils_ROMAN import _pair_label, load_gt_data_ROMAN, load_est_data_ROMAN

def _print_metrics(metrics_dictionary: dict) -> None:
    print("RMS ATE: ", metrics_dictionary['APE']['translation_part']['rmse'])
    print("RMS RTE: ", metrics_dictionary['RPE']['translation_part']['rmse'])
    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary['APE']['rotation_angle_deg']['rmse'])
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary['RPE']['rotation_angle_deg']['rmse'])

def calculate_merged_ate(dataset_name: str, method: str, robot_names: list, visualize: bool = False, do_individual_calcs: bool = False) -> float:
    robot0_name = robot_names[0]
    robot1_name = robot_names[1]

    est_data_lst: list[OdometryData] = load_est_data_ROMAN(dataset_name, method, robot_names)
    gt_data_lst: list[OdometryData] = load_gt_data_ROMAN(dataset_name, robot_names)
    est_data_robot0, est_data_robot1 = est_data_lst
    gt_data_robot0, gt_data_robot1 = gt_data_lst

    # Calculate individual RMS ATE
    if do_individual_calcs:
        # TODO: Need to make start and end times match before individual RMS ATE as well;
        # if we ever use those results in a paper.

        print("=========== Individual Trajectory", robot0_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot0, est_data_robot0, max_diff=0.1, visualize=False)
        _print_metrics(metrics_dictionary)

        print("\n=========== Individual Trajectory", robot1_name, "for dataset: ", dataset_name, method, "============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_robot1, est_data_robot1, max_diff=0.1, visualize=False)
        _print_metrics(metrics_dictionary)

    # Make the timestamps match and then concatenate
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)
    est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
    gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

    # Calculate RMS ATE, among other metrics
    print("\n========== Merged Trajectories for dataset: ", dataset_name, method, "_".join(robot_names), "==========")
    metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=False)
    _print_metrics(metrics_dictionary)

    if visualize:
        # Seperate the aligned trajectories into their single-robot forms
        gt_data_align_list = PathData.seperate_PathData(gt_data_lst, gt_data_align)
        gt_data_align_robot0 = gt_data_align_list[0]
        gt_data_align_robot1 = gt_data_align_list[1]

        est_data_align_list = PathData.seperate_PathData(est_data_lst, est_data_align)
        est_data_align_robot0 = est_data_align_list[0]
        est_data_align_robot1 = est_data_align_list[1]

        # Get environment image path
        user = getpass.getuser()
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

        # Load LC data with display names so names match nameList
        names_override_display = {chr(97 + i): name_map[rn] for i, rn in enumerate([robot0_name, robot1_name])}
        _, lc_data_inlier = load_LC_data_ROMAN(dataset_name, method, [robot0_name, robot1_name],
                                               only_inter_lc=True, names_override=names_override_display)
        gt_dict_display = {name_map[robot0_name]: gt_data_robot0, name_map[robot1_name]: gt_data_robot1}
        lc_errors_viz = lc_data_inlier.calculate_errors(gt_dict_display)

        pair_label = _pair_label(robot0_name, robot1_name)
        base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures') / dataset_name
        traj_dir = base_dir / 'traj'
        traj_lc_dir = base_dir / 'traj_lc'
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_lc_dir.mkdir(parents=True, exist_ok=True)

        # Plot the results in 2D (Configuration for Figure 10)
        dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
        isGTList =  [                False,                 True,                  False,                  True]
        nameList =  [name_map[robot0_name], name_map[robot0_name], name_map[robot1_name], name_map[robot1_name]]
        colorList = [robot_name_to_color[name] for name in nameList]
        PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=2.0, show_grid=True,
                           background_image_path=image_path, background_image_x_edge=x_edge,
                           save_path=str(traj_dir / f'traj_{pair_label}_{method}.pdf'))

        # Plot estimated trajectories with LC overlay (no background, no GT)
        est_dataList =  [est_data_align_robot0,       est_data_align_robot1]
        est_isGTList =  [               False,                        False]
        est_nameList =  [name_map[robot0_name], name_map[robot1_name]]
        est_colorList = [robot_name_to_color[name] for name in est_nameList]
        PathData.visualize_2D(est_dataList, est_isGTList, est_colorList, est_nameList, no_background=True, line_width=1.0, show_grid=True,
                           loop_closure_data=lc_data_inlier, lc_errors=lc_errors_viz, lc_line_width=2.0, lc_errors_vmax=50.0,
                           title=f"{method} LC overlaid on trajectory",
                           save_path=str(traj_lc_dir / f'traj_lc_{pair_label}_{method}.pdf'))

        # Configuration for Figure 2
        # PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=4.0, show_grid=False,
        #                     background_image_path=image_path, background_image_x_edge=x_edge, legend=False, no_border=True,
        #                     save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')

    return metrics_dictionary['APE']['translation_part']['rmse']


def main():
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    robot_pairs = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN", "ROMAN_NM", "MG_TS"]
    dataset_name = "V2.4.C"

    tasks = [(dataset_name, run_name, list(pair), True)
             for pair in robot_pairs
             for run_name in run_names]

    with Pool() as pool:
        results = pool.starmap(calculate_merged_ate, tasks)

    table_data: dict[str, dict[str, float]] = {run: {} for run in run_names}
    for (_, run_name, pair, *_), ate in zip(tasks, results):
        col = _pair_label(*pair)
        table_data[run_name][col] = ate

    cols = [_pair_label(*p) for p in robot_pairs]

    inlier_lc_total: dict[str, dict[str, int]] = {run: {} for run in run_names}
    for pair in robot_pairs:
        col = _pair_label(*pair)
        gt_list = load_gt_data_ROMAN(dataset_name, list(pair))
        gt_dict = {name: gt for name, gt in zip(pair, gt_list)}
        for run_name in run_names:
            merged_lc, merged_lc_inlier = load_LC_data_ROMAN(dataset_name, run_name, list(pair), only_inter_lc=True)
            _, inlier_errs = calculate_LC_errors_ROMAN(merged_lc, merged_lc_inlier, gt_dict)
            inlier_lc_total[run_name][col] = len(inlier_errs['translation_errors'])

    RUN_DISPLAY_NAMES = {
        "ROMAN": "ROMAN",
        "ROMAN_Deduplication": "ROMAN w/o duplicate LC",
        "ROMAN_NM":   "NM + ROMAN",
        "ROMAN_NM_POA_Triplet": "NM + ROMAN + Triplet POA",
        "MG_TS": "NM + MG (Above but with Global MG)",
        "MG_TS_Duplication": "NM + MG (Above but with Dup. LC)",
        "MG_TS_<Old_Version>": "NM + MG (Two Stage - Reworked 4)",
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

    base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures') / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)
    save_styled_tables(
        dfs,
        str(base_dir / 'ate_table.pdf'),
        cell_is_red=lambda s: s == "---" or float(s) > 20)


if __name__ == "__main__":
    main()

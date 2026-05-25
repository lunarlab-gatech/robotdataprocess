import getpass
import itertools
import re
import sys
from pathlib import Path
import pandas as pd
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.visualization import save_styled_tables


def _pair_label(name_a: str, name_b: str) -> str:
    def abbrev(n):
        m = re.match(r'([A-Za-z]+)(\d+)', n)
        return (m.group(1)[0].upper() + m.group(2)) if m else n
    return abbrev(name_a) + abbrev(name_b)

def calculate_LC_errors_ROMAN(dataset_name: str, run_name: str, robot_names: List, only_inter_lc: bool = False):

    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + run_name + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    pair_fn = itertools.combinations if only_inter_lc else itertools.combinations_with_replacement

    # Load loop closure data for every robot pair in this run
    lc_data_list = []
    lc_inlier_data_list = []
    for name_a, name_b in pair_fn(robot_names, 2):

        # Load the Loop Closures that are fed to Kimera-RPGO (removes loops to same exact vertex but keeps those LC that are consecutive on same robot)
        lc_data = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' / 'dense' / 'odom_and_lc.g2o',
            run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
            names_override={"a": name_a, "b": name_b})
        lc_data_list.append(lc_data)

        # Load the inliers detected by Kimera-RPGO
        letter_a = chr(97 + robot_names.index(name_a))
        letter_b = chr(97 + robot_names.index(name_b))
        if name_a == name_b:
            g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else:
            g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' / g2o_filename,
                                                  run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
                                                  names_override={"a": name_a, "b": name_b})
        lc_inlier_data_list.append(lc_data_inlier)

    merged_lc = LoopClosureData.merge(lc_data_list)
    merged_lc_inlier = LoopClosureData.merge(lc_inlier_data_list)

    # Load the GT data for all robots
    dataset_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_name + '/data/'
    gt_data_dict: dict[str, OdometryData] = {}
    for rn in robot_names:
        gt_data = OdometryData.from_txt(dataset_path + rn + '/pose_world_frame.txt', 'world', 'robot', CoordinateFrame.NED, False)
        gt_data.to_coordinate_frame(CoordinateFrame.FLU)
        gt_data_dict[rn] = gt_data

    return (merged_lc.calculate_errors(gt_data_dict),
            merged_lc_inlier.calculate_errors(gt_data_dict))

def main():
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    robot_pairs = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN_NM", "MG_SS_3", "MG_SS_3_POA"]
    dataset_name = "V2.4.C"

    table_data: dict[str, dict[str, dict]] = {run: {} for run in run_names}
    table_data_inlier: dict[str, dict[str, dict]] = {run: {} for run in run_names}

    for pair in robot_pairs:
        col = _pair_label(*pair)

        errors_list = []
        labels_list = []
        group_indices = []
        for i, run_name in enumerate(run_names):
            all_errs, inlier_errs = calculate_LC_errors_ROMAN(dataset_name, run_name, list(pair), only_inter_lc=False)
            errors_list.append(all_errs)
            labels_list.append(run_name)
            group_indices.append(i)
            errors_list.append(inlier_errs)
            labels_list.append(run_name + " [Inliers]")
            group_indices.append(i)

        _, stats = LoopClosureData.visualize_error_scatter(
            errors_list, labels_list, group_indices=group_indices,
            max_rotation_frac=1.0, max_translation_frac=1.0,
            trans_err_in_target=1.0, show_plots=False, rot_err_in_target=5.0,
            save_path=f'/home/dbutterfield3/Research/robotdataprocess/lc_{col}.pdf')

        for i, run_name in enumerate(run_names):
            table_data[run_name][col] = stats[2 * i]
            table_data_inlier[run_name][col] = stats[2 * i + 1]

    RUN_DISPLAY_NAMES = {
        "ROMAN_NM":   "NM + ROMAN",
        "MG_TS_2-4":  "NM + MG (Two Stage - 2/4 req)",
        "MG_TS_3-4":  "NM + MG (Two Stage - 3/4 req)",
        "MG_SS_3":    "NM + MG (Single Stage - 3 req)",
        "MS_SS_3_POA":"NM + MG (SS3) + POA"
    }

    def make_df(data, key, fmt):
        return pd.DataFrame(
            {RUN_DISPLAY_NAMES.get(run, run): {col: fmt(data[run][col][key]) for col in data[run]}
             for run in run_names}
        ).T

    def make_rank_df(data, key):
        return pd.DataFrame(
            {RUN_DISPLAY_NAMES.get(run, run): {col: float(data[run][col][key]) for col in data[run]}
             for run in run_names}
        ).T

    def make_combined_df(data):
        def fmt(s): return f"{s['num_successful_loop_closures']}/{s['num_loop_closures']}"
        return pd.DataFrame(
            {RUN_DISPLAY_NAMES.get(run, run): {col: fmt(data[run][col]) for col in data[run]}
             for run in run_names}
        ).T

    dfs = [
        ("LC Success Rate %",
         make_df(table_data, "success_rate", lambda x: f"{x:.1f}%"),
         [make_rank_df(table_data, "success_rate")]),
        ("LC Successful / Total",
         make_combined_df(table_data),
         [make_rank_df(table_data, "num_successful_loop_closures"),
          make_rank_df(table_data, "num_loop_closures")]),
        ("Inlier LC Success Rate %",
         make_df(table_data_inlier, "success_rate", lambda x: f"{x:.1f}%"),
         [make_rank_df(table_data_inlier, "success_rate")]),
        ("Inlier LC Successful / Total",
         make_combined_df(table_data_inlier),
         [make_rank_df(table_data_inlier, "num_successful_loop_closures"),
          make_rank_df(table_data_inlier, "num_loop_closures")]),
    ]

    save_styled_tables(dfs, '/home/dbutterfield3/Research/robotdataprocess/lc_tables.pdf')

if __name__ == "__main__":
    main()
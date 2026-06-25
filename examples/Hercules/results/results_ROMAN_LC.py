from collections import Counter
import getpass
import itertools
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pandas as pd
from robotdataprocess import LoopClosureData
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from utils.visualization import save_styled_tables
from utils_ROMAN import _pair_label, load_gt_data_ROMAN

def load_LC_data_ROMAN(dataset_name: str, run_name: str, robot_names: List, only_inter_lc: bool = False,
                       names_override: dict = None):
    """
    Load LC data for a ROMAN run.

    Args:
        names_override: If provided, maps g2o character keys ('a', 'b', ...) to robot names used in
            the returned LoopClosureData. Defaults to the original robot_names. Pass display-name
            overrides when LC names must match a visualize_2D nameList.

    Returns:
        (merged_lc, merged_lc_inlier)
    """
    user = getpass.getuser()
    run_folder = Path('/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + run_name + '/' + \
                      (robot_names[0] + '_' + robot_names[1]))

    pair_fn = itertools.combinations if only_inter_lc else itertools.combinations_with_replacement

    effective_override = names_override if names_override is not None else \
        {chr(97 + i): name for i, name in enumerate(robot_names)}

    # odom_and_lc.g2o already contains all robot pairs — load it once to avoid
    # tripling the count when iterating over combinations_with_replacement.
    merged_lc = LoopClosureData.from_g2o(
        run_folder / 'offline_rpgo' / 'dense' / 'odom_and_lc.g2o',
        run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
        names_override=effective_override)
    if only_inter_lc:
        merged_lc.prune_intra_robot_loop_closures()

    # Load the per-pair inlier g2o files (these are pair-specific)
    lc_inlier_data_list = []
    for name_a, name_b in pair_fn(robot_names, 2):
        letter_a = chr(97 + robot_names.index(name_a))
        letter_b = chr(97 + robot_names.index(name_b))
        if name_a == name_b:
            g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else:
            g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        lc_data_inlier = LoopClosureData.from_g2o(run_folder / 'offline_rpgo' / g2o_filename,
                                                  run_folder / 'offline_rpgo' / 'dense' / 'odom_all.time.txt',
                                                  names_override=effective_override)
        lc_inlier_data_list.append(lc_data_inlier)

    merged_lc_inlier = LoopClosureData.merge(lc_inlier_data_list)

    # Print information on duplicates
    # merged_lc.print_duplicate_info(f"merged_lc ({dataset_name}/{run_name})")
    # merged_lc_inlier.print_duplicate_info(f"merged_lc_inlier ({dataset_name}/{run_name})")

    return merged_lc, merged_lc_inlier

def calculate_LC_errors_ROMAN(merged_lc: LoopClosureData, merged_lc_inlier: LoopClosureData, gt_data_dict: dict):
    return (merged_lc.calculate_errors(gt_data_dict), merged_lc_inlier.calculate_errors(gt_data_dict))

def main(): 
    all_robots = ["Husky1", "Husky2", "Drone1", "Drone2"]
    robot_pairs = list(itertools.combinations(all_robots, 2))
    run_names = ["ROMAN", "ROMAN_NM", "MG_TS"] # "ROMAN_NM_POA_Pair", "ROMAN_NM_POA_Triplet"] # "MG_TS", "MG_TS_POA", "MG_TS_2-4", "MG_TS_3-4", "MG_SS_3",
    dataset_name = "V2.4.C"

    base_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures') / dataset_name
    lc_dir = base_dir / 'lc'
    lc_sr_dir = base_dir / 'lc_success_rate'
    lc_dir.mkdir(parents=True, exist_ok=True)
    lc_sr_dir.mkdir(parents=True, exist_ok=True)

    table_data: dict[str, dict[str, dict]] = {run: {} for run in run_names}
    table_data_inlier: dict[str, dict[str, dict]] = {run: {} for run in run_names}

    for pair in robot_pairs:
        col = _pair_label(*pair)

        errors_list = []
        labels_list = []
        group_indices = []
        gt_list = load_gt_data_ROMAN(dataset_name, list(pair))
        gt_dict = {name: gt for name, gt in zip(pair, gt_list)}
        for i, run_name in enumerate(run_names):
            merged_lc, merged_lc_inlier = load_LC_data_ROMAN(dataset_name, run_name, list(pair), only_inter_lc=True)
            all_errs, inlier_errs = calculate_LC_errors_ROMAN(merged_lc, merged_lc_inlier, gt_dict)
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
            save_path=str(lc_dir / f'lc_{col}.pdf'))

        # LoopClosureData.visualize_error_AUC(
        #     errors_list, labels_list, show_plots=False,
        #     save_path=str(lc_dir / f'lc_curve_{col}.pdf'))

        fig_sr = LoopClosureData.visualize_success_rate(
            errors_list[::2], labels_list[::2], show_plots=False, max_translation_frac=0.01, max_rotation_frac=0.035,
            include_rate_plots=False)
        fig_sr.savefig(str(lc_sr_dir / f'lc_{col}_success_rate.pdf'))
        plt.close(fig_sr)

        for i, run_name in enumerate(run_names):
            table_data[run_name][col] = stats[2 * i]
            table_data_inlier[run_name][col] = stats[2 * i + 1]

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

    save_styled_tables(dfs, str(base_dir / 'lc_tables.pdf'))

if __name__ == "__main__":
    main()
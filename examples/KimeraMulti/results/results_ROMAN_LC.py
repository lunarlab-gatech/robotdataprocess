import getpass
import itertools
import numpy as np
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

DATASET_SEQUENCE = {"1014": "Outdoor", "1208": "Hybrid", "1207": "Tunnel"}


def calculate_LC_errors_ROMAN(run_path: str, robot_names: List, dataset_number: str,
                               all_error_dicts: list, all_inlier_masks: list):

    user = getpass.getuser()
    run_folder = Path('/home/') / user / 'Research' / run_path

    # Symbol-to-name mapping: 'a' -> robot_names[0], 'b' -> robot_names[1], etc.
    g2o_symbol_to_name = {chr(97 + i): robot_names[i] for i in range(len(robot_names))}

    # Load and label loop closure data for every robot pair in this run
    lc_list = []
    for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):

        lc_data = LoopClosureData.from_json(run_folder / 'align' / (name_a + '_' + name_b) / 'align.json')

        letter_a = chr(97 + robot_names.index(name_a))
        letter_b = chr(97 + robot_names.index(name_b))
        lc_data_inlier = None
        if name_a == name_b: g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else: g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        try:
            lc_data_inlier = LoopClosureData.from_g2o(
                run_folder / 'offline_rpgo' / g2o_filename,
                run_folder / 'offline_rpgo' / 'sparse' / 'odom_all.time.txt',
                names_override=g2o_symbol_to_name)
        except FileNotFoundError as e:
            print(f"Missing inliers for pair {name_a}_{name_b} in {run_folder}", e)

        lc_data.round_timestamps(4)
        if lc_data_inlier:
            lc_data_inlier.round_timestamps(4)
            lc_data.label_inliers_via_other_LoopClosureData(lc_data_inlier)

        lc_list.append(lc_data)

    # Merge all pairs into one LC object for this run
    merged_lc = LoopClosureData.merge(lc_list)

    # Load GT data for all robots
    dataset_path = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data' / 'ground_truth' / dataset_number
    gt_data_dict: dict[str, OdometryData] = {}
    for rn in robot_names:
        gt_data_dict[rn] = OdometryData.from_csv(dataset_path / (rn + '_gt_odom.csv'),
                                                  'world', 'robot', CoordinateFrame.FLU, True, None, ts_in_ns=True)

    # Calculate errors and collect results
    all_error_dicts.append(merged_lc.calculate_errors(gt_data_dict))
    if hasattr(merged_lc, 'detected_inliers'):
        all_inlier_masks.append(merged_lc.detected_inliers)
    else:
        all_inlier_masks.append(np.zeros(merged_lc.num_loop_closures, dtype=bool))


def main():
    user = getpass.getuser()
    #robot_names = ["acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth"]
    #dataset_number = "1014"
    robot_names = ["acl_jackal"]
    dataset_number = "1207"
    sequence = DATASET_SEQUENCE[dataset_number]
    robots_str = "_".join(robot_names)

    run_paths = [
        f"roman/kimera_multi_output/Easy/{sequence}/{robots_str}",
        f"ROMAN_DEVEL/results/Kimera-Multi_ROMAN_NM/Easy/{sequence}/{robots_str}",
    ]

    errors_list = []
    inliers_list = []
    for run_path in run_paths:
        calculate_LC_errors_ROMAN(run_path, robot_names, dataset_number, errors_list, inliers_list)

    LoopClosureData.visualize_error_scatter(errors_list, run_paths, inliers_list, max_rotation_frac=1.0,
                                            max_translation_frac=1.0, trans_err_in_target=1.0, show_plots=False,
                                            rot_err_in_target=5.0, save_path='/home/dbutterfield3/Research/robotdataprocess/lc_fig.pdf')


if __name__ == "__main__":
    main()

import getpass
import itertools
from pathlib import Path
from robotdataprocess import LoopClosureData, OdometryData, CoordinateFrame
from typing import List

DATASET_SEQUENCE = {"1014": "Outdoor", "1208": "Hybrid", "1207": "Tunnel"}


def calculate_LC_errors_ROMAN(run_path: str, robot_names: List, dataset_number: str):

    user = getpass.getuser()
    run_folder = Path('/home/') / user / 'Research' / run_path

    # Symbol-to-name mapping: 'a' -> robot_names[0], 'b' -> robot_names[1], etc.
    g2o_symbol_to_name = {chr(97 + i): robot_names[i] for i in range(len(robot_names))}

    # Load loop closure data for every robot pair in this run
    lc_list = []
    inlier_lc_list = []
    for name_a, name_b in itertools.combinations_with_replacement(robot_names, 2):

        lc_data = LoopClosureData.from_json(run_folder / 'align' / (name_a + '_' + name_b) / 'align.json')
        lc_list.append(lc_data)

        letter_a = chr(97 + robot_names.index(name_a))
        letter_b = chr(97 + robot_names.index(name_b))
        if name_a == name_b:
            g2o_filename = f'inlier_lc_intra_{letter_a}.g2o'
        else:
            g2o_filename = f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        try:
            lc_data_inlier = LoopClosureData.from_g2o(
                run_folder / 'offline_rpgo' / g2o_filename,
                run_folder / 'offline_rpgo' / 'sparse' / 'odom_all.time.txt',
                names_override=g2o_symbol_to_name)
            inlier_lc_list.append(lc_data_inlier)
        except FileNotFoundError as e:
            print(f"Missing inliers for pair {name_a}_{name_b} in {run_folder}", e)

    # Merge all pairs into one LC object for this run
    merged_lc = LoopClosureData.merge(lc_list)
    merged_lc_inlier = LoopClosureData.merge(inlier_lc_list)

    # Load intermediate LCs from odom_and_lc.g2o (odometry edges filtered out by from_g2o)
    try:
        merged_lc_intermediate = LoopClosureData.from_g2o(
            run_folder / 'offline_rpgo' / 'odom_and_lc.g2o',
            run_folder / 'offline_rpgo' / 'sparse' / 'odom_all.time.txt',
            names_override=g2o_symbol_to_name)
    except FileNotFoundError as e:
        print(f"Missing odom_and_lc.g2o in {run_folder}", e)
        merged_lc_intermediate = LoopClosureData.merge([])

    # Load GT data for all robots
    dataset_path = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data' / 'ground_truth' / dataset_number
    gt_data_dict: dict[str, OdometryData] = {}
    for rn in robot_names:
        gt_data_dict[rn] = OdometryData.from_csv(dataset_path / (rn + '_gt_odom.csv'),
                                                  'world', 'robot', CoordinateFrame.FLU, True, None, ts_in_ns=True)

    for lc in (merged_lc, merged_lc_intermediate, merged_lc_inlier):
        lc.calculate_errors(gt_data_dict)
        lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

    return merged_lc, merged_lc_intermediate, merged_lc_inlier


def main():
    # NOTE: THIS FILE IS STILL A WORK IN PROGRESS.

    #robot_names = ["acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek"]
    robot_names = ["acl_jackal"]
    dataset_number = "1207"
    sequence = DATASET_SEQUENCE[dataset_number]
    difficulty = "Easy"
    robots_str = "_".join(robot_names)

    run_paths = [
        f"roman/results/Kimera-Multi_ROMAN/{difficulty}/{sequence}/{robots_str}",
        f"ROMAN_DEVEL/results/Kimera-Multi_ROMAN_NM/{difficulty}/{sequence}/{robots_str}",
        f"ROMAN_DEVEL/results/Kimera-Multi_MG/{difficulty}/{sequence}/{robots_str}",
    ]
    run_names = ["MG"] # ["ROMAN", "ROMAN + NM", "MG"]

    lc_data_list = []
    labels_list = []
    group_indices = []
    n_runs = len(run_paths)
    for i, (run_path, run_name) in enumerate(zip(run_paths, run_names)):
        all_lc, intermediate_lc, inlier_lc = calculate_LC_errors_ROMAN(run_path, robot_names, dataset_number)

        lc_data_list.append(all_lc)
        labels_list.append(run_name)
        group_indices.append(i)

        lc_data_list.append(intermediate_lc)
        labels_list.append(run_name + " [Intermediate Step]")
        group_indices.append(i + 1)

        lc_data_list.append(inlier_lc)
        labels_list.append(run_name + " [Inliers]")
        group_indices.append(i + 2)

    LoopClosureData.visualize_error_scatter(lc_data_list, labels_list, group_indices=group_indices,
                                            max_rotation_frac=1.0, max_translation_frac=1.0, show_plots=False,
                                            save_path='/home/dbutterfield3/Research/robotdataprocess/' + dataset_number + "-" + difficulty + '-' + robots_str +'.pdf')


if __name__ == "__main__":
    main()

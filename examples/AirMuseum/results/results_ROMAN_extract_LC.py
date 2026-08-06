import copy
import itertools
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from utils.ROMAN import load_LC_data_ROMAN, LCFilterMode
from results_ROMAN import load_gt_data_ROMAN

def main():
    dataset_prefix = "airmuseum"
    dataset_name = "Scenario5"
    pair = ["robotA", "robotB"]
    run_names = ["ROMAN_O", "MG_NONM"]
    out_dir = Path(__file__).parent.parent.parent.parent / 'figures' / dataset_prefix / dataset_name / 'ALL'
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_name_to_chars_mapping: dict = {
        "drone": "D",
        "robotA": "RA",
        "robotB": "RB",
        "robotC": "RC"
    }

    gt_list = load_gt_data_ROMAN(dataset_name, pair)
    gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

    merged_lc_by_run: Dict[str, LoopClosureData] = {}
    for run_name in run_names:
        merged_lc, _ = load_LC_data_ROMAN(dataset_prefix, dataset_name, run_name, pair, lc_filter=LCFilterMode.ALL)
        merged_lc.calculate_errors(gt_dict)
        merged_lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)
        merged_lc_by_run[run_name] = merged_lc

        successful_lc = copy.deepcopy(merged_lc)
        successful_lc._prune_by_mask(successful_lc.results.successful)

        out_path = out_dir / f'successful_lc_{robot_name_to_chars_mapping[pair[0]]}-{robot_name_to_chars_mapping[pair[1]]}_{run_name}.json'
        successful_lc.to_json(out_path)
        print(f"{run_name}: {successful_lc.num_loop_closures} successful loop closures written to {out_path}")

    for self_run_name, other_run_name in itertools.permutations(run_names, 2):
        print(f"\n--- Successful in {self_run_name} but not {other_run_name} ---")
        merged_lc_by_run[self_run_name].print_successful_lc_diff(
            merged_lc_by_run[other_run_name], self_run_name, other_run_name)

if __name__ == "__main__":
    main()

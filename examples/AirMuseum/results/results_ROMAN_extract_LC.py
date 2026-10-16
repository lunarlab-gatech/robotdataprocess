import copy
import itertools
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from robotdataprocess.data_types.SLAMData import SLAMData
from results_ROMAN import load_gt_data_ROMAN

def main():
    roman_root = Path('/home/dbutterfield3/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": True}
    dataset_prefix = "airmuseum"
    dataset_seq = "Scenario5"
    pair = sorted(["robotA", "robotB"])
    run_names = ["ROMAN_O", "MG_NONM"]
    out_dir = Path(__file__).parent.parent.parent.parent / 'figures' / dataset_prefix / dataset_seq / 'ALL'
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_name_to_chars_mapping: dict = {
        "drone": "D",
        "robotA": "RA",
        "robotB": "RB",
        "robotC": "RC"
    }

    gt_list = load_gt_data_ROMAN(dataset_seq, pair)
    gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

    merged_lc_by_run: Dict[str, LoopClosureData] = {}
    for run_name in run_names:
        system_params = SLAMData.load_system_params(roman_root, dataset_prefix, dataset_seq, run_name)
        merged_lc, _ = SLAMData.load_LC_data(roman_root, system_params, pair, critical_invocation_params)
        merged_lc.calculate_errors(gt_dict)
        merged_lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)
        merged_lc_by_run[run_name] = merged_lc

        successful_lc = copy.deepcopy(merged_lc)
        successful_lc._prune_by_mask(successful_lc.results.successful)

        out_path = out_dir / f'successful_lc_{robot_name_to_chars_mapping[pair[0]]}-{robot_name_to_chars_mapping[pair[1]]}_{run_name}.json'
        successful_lc.to_json(out_path)
        print(f"{run_name}: {successful_lc.num_loop_closures} successful loop closures written to {out_path}")

    for self_run_name, other_run_name in itertools.permutations(run_names, 2):
        print(f"\n--- LC successful diff: {self_run_name} vs {other_run_name} ---")
        merged_lc_by_run[self_run_name].print_successful_lc_diff(
            merged_lc_by_run[other_run_name], self_run_name, other_run_name)

if __name__ == "__main__":
    main()

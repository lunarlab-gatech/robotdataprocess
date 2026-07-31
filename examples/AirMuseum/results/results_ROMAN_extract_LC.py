import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from utils.ROMAN import load_LC_data_ROMAN, LCFilterMode
from results_ROMAN import load_gt_data_ROMAN

def main():
    dataset_prefix = "airmuseum"
    dataset_name = "Scenario5"
    pair = ["drone", "robotA"]
    run_names = ["ROMAN_O", "MG_NONM"]
    out_dir = Path(__file__).parent.parent.parent.parent / 'figures' / dataset_prefix / dataset_name / 'ALL'

    gt_list = load_gt_data_ROMAN(dataset_name, pair)
    gt_dict = {name: gt for name, gt in zip(pair, gt_list)}

    for run_name in run_names:
        merged_lc, _ = load_LC_data_ROMAN(dataset_prefix, dataset_name, run_name, pair, lc_filter=LCFilterMode.ALL)
        merged_lc.calculate_errors(gt_dict)
        merged_lc.label_successful(trans_err_in_target=1.0, rot_err_in_target=5.0)

        successful_lc = copy.deepcopy(merged_lc)
        successful_lc._prune_by_mask(successful_lc.results.successful)

        out_path = out_dir / f'successful_lc_D-RA_{run_name}.json'
        successful_lc.to_json(out_path)
        print(f"{run_name}: {successful_lc.num_loop_closures} successful loop closures written to {out_path}")

if __name__ == "__main__":
    main()

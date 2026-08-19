import copy
import itertools
from multiprocessing import Pool
import numpy as np
from pathlib import Path
import sys
import tqdm
from typing import Any, Dict, List
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.ROMAN import calculate_merged_ate, load_LC_data_ROMAN, load_system_params_ROMAN, LCFilterMode
from results_ROMAN import load_gt_data_ROMAN

def _calculate_merged_ate_star(task: Dict[str, Any]):
    """
    Runs one calculate_merged_ate task (see main's tasks list) -- Pool.imap only supports
    single-argument workers. Re-raises any failure with the specific base config, overrides,
    and robot pair that caused it, since the original exception alone gives no indication of
    which of potentially hundreds of pooled tasks actually failed.
    """
    try:
        return calculate_merged_ate(*task["args"])
    except Exception as e:
        raise RuntimeError(f"calculate_merged_ate failed for base_config='{task['base_config']}', "
                           f"overrides={task['overrides']}, robot_pair={task['pair']}") from e

def load_sweep_data(sweep_path: Path) -> Dict[str, Any]:
    """ Loads a W&B sweep config YAML, raising if it isn't a "method: grid" sweep. """
    with open(sweep_path) as f:
        sweep_data = yaml.safe_load(f)
    if sweep_data.get('method') != 'grid':
        raise ValueError(f"{sweep_path} is a '{sweep_data.get('method')}' sweep, not 'grid' -- "
                         f"only grid sweeps can be enumerated exhaustively without the wandb API.")
    return sweep_data

def sweep_param_grid(sweep_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enumerate every parameter combination in a "method: grid" sweep config (see
    load_sweep_data), as a list of dicts mapping dotted param key (e.g.
    "submap_align_params.num_req_assoc") to value. Grid sweeps enumerate their full cartesian
    product exhaustively, so this reproduces the exact set of configs `wandb agent` would have
    run, without needing the wandb API.
    """
    param_names = list(sweep_data['parameters'].keys())
    value_lists = [sweep_data['parameters'][name]['values'] for name in param_names]
    return [dict(zip(param_names, combo)) for combo in itertools.product(*value_lists)]

def check_no_overlapping_params(sweep_data_by_name: Dict[str, Dict[str, Any]]) -> None:
    """
    Raises if any two sweeps override the same parameter key -- otherwise pooling their grids
    (see main) would silently duplicate that parameter's override across the pooled task list,
    double-counting whatever (method, config, pair) combinations depend on it.
    """
    owner_by_param: Dict[str, str] = {}
    for sweep_name, sweep_data in sweep_data_by_name.items():
        for param in sweep_data['parameters']:
            if param in owner_by_param:
                raise ValueError(f"Sweeps '{owner_by_param[param]}' and '{sweep_name}' both override "
                                 f"'{param}' -- remove the overlap before pooling their grids.")
            owner_by_param[param] = sweep_name

def apply_overrides(system_params, overrides: Dict[str, Any]):
    """
    Applies dotted-key overrides (e.g. {"submap_align_params.num_req_assoc": 3}) to a deep copy
    of system_params, matching run_slam.py's own wandb override-application convention. Only
    supports top-level keys (e.g. "seed") or one level of nesting (e.g.
    "submap_align_params.num_req_assoc") -- not deeper paths like "a.b.c".

    Raises:
        AttributeError: If a key names a parameter (or parent) that doesn't exist on
            system_params, rather than silently creating a new attribute.
    """
    system_params = copy.deepcopy(system_params)
    for key, value in overrides.items():
        parts = key.split('.')
        if len(parts) == 1:
            if not hasattr(system_params, parts[0]):
                raise AttributeError(f"system_params has no parameter '{parts[0]}' (override key '{key}')")
            setattr(system_params, parts[0], value)
        else:
            parent_name, param_name = parts
            if not hasattr(system_params, parent_name):
                raise AttributeError(f"system_params has no parameter group '{parent_name}' (override key '{key}')")
            parent_obj = getattr(system_params, parent_name)
            if not hasattr(parent_obj, param_name):
                raise AttributeError(f"{parent_name} has no parameter '{param_name}' (override key '{key}')")
            setattr(parent_obj, param_name, value)
    return system_params

def main():
    """
    Report alignment success/failure, on one robot pair, across every config in the given
    sweeps, for each of the hardcoded methods below. Each method starts from its own base
    config (as in results_ROMAN.py), then gets every parameter combination from every named
    sweep applied on top of it.

    A (method, sweep config) combination is a success if its Merged RMS ATE is below
    ate_threshold_m AND it has at least one inlier inter-robot loop closure; otherwise a failure.
    Missing result directories (a sweep that was never actually run for a given method) raise
    FileNotFoundError rather than being silently skipped.
    """
    roman_root = Path('/home/dbutterfield3/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": True}
    dataset_prefix = "airmuseum"
    dataset_name = "Scenario5"
    robot_pair = ["drone", "robotB"]
    methods = ["MG"]
    sweep_names = ["sweep_harder_params"]
    ate_threshold_m = 10.0
    trans_err_in_target = 1.0
    rot_err_in_target = 5.0

    gt_dict = dict(zip(robot_pair, load_gt_data_ROMAN(dataset_name, robot_pair)))

    sweeps_dir = roman_root / "research" / "AirMuseum" / "sweeps"

    sweep_data_by_name = {name: load_sweep_data(sweeps_dir / f'{name}.yaml') for name in sweep_names}
    check_no_overlapping_params(sweep_data_by_name)
    overrides_list = [o for sweep_data in sweep_data_by_name.values() for o in sweep_param_grid(sweep_data)]

    # Build every (method, swept system_params) task up front, then run the ATE calculations
    # (the expensive part) in parallel, exactly like run_ROMAN_evaluation does.
    tasks = []
    for method in methods:
        base_system_params = load_system_params_ROMAN(roman_root, dataset_prefix, dataset_name, method)
        base_config = f"{dataset_prefix}_{dataset_name}_{method}.yaml"
        for overrides in overrides_list:
            swept_system_params = apply_overrides(base_system_params, overrides)
            tasks.append({
                "args": (roman_root, swept_system_params, dataset_prefix, dataset_name, method,
                        robot_pair, critical_invocation_params, load_gt_data_ROMAN),
                "method": method,
                "base_config": base_config,
                "overrides": overrides,
                "pair": robot_pair,
            })

    with Pool() as pool:
        results = list(tqdm.tqdm(pool.imap(_calculate_merged_ate_star, tasks), total=len(tasks),
                                 desc="Computing ATE...", unit=" configs"))

    counts = {method: {"success": 0, "failure": 0} for method in methods}
    for task, result in zip(tasks, results):
        method = task["method"]
        system_params, robot_names = task["args"][1], task["args"][5]
        ate = result.merged_metrics.APE.translation_part.rmse

        lc_all, lc_inlier = load_LC_data_ROMAN(roman_root, system_params, dataset_prefix, dataset_name, robot_names,
                                         critical_invocation_params, lc_filter=LCFilterMode.ONLY_INTER_LC)
        num_inlier_inter_lc = lc_inlier.num_loop_closures

        lc_all.calculate_errors(gt_dict)
        lc_all.label_successful(trans_err_in_target, rot_err_in_target)
        lc_inlier.calculate_errors(gt_dict)
        lc_inlier.label_successful(trans_err_in_target, rot_err_in_target)
        num_successful_inlier_inter_lc = int(np.sum(lc_inlier.results.successful))
        num_successful_inter_lc = int(np.sum(lc_all.results.successful))
        num_total_inter_lc = lc_all.num_loop_closures

        success = ate < ate_threshold_m and num_inlier_inter_lc >= 1
        counts[method]["success" if success else "failure"] += 1
        print(f"[{method}] {robot_names}: ATE={ate:.2f}m, inlier LC={num_successful_inlier_inter_lc}/{num_inlier_inter_lc} "
              f"all LC={num_successful_inter_lc}/{num_total_inter_lc} "
              f"-> {'SUCCESS' if success else 'FAILURE'}")

    print("\n=== Summary ===")
    for method in methods:
        s, f = counts[method]["success"], counts[method]["failure"]
        print(f"{method}: {s} successes, {f} failures (of {s + f})")

if __name__ == "__main__":
    main()

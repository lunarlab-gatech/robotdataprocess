import getpass
from pathlib import Path
import re
from robotdataprocess import OdometryData, CoordinateFrame, PathData
from scipy.spatial.transform import Rotation as R

DATASET_SEQUENCE = {"1014": "Outdoor", "1208": "Hybrid", "1207": "Tunnel"}

def _parse_map_runtimes(path: Path, robot_names: set) -> dict:
    """Parse map/runtime.txt, returning {robot_name: seconds} for robots in robot_names."""
    runtimes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ': ' not in line:
                continue
            name, val = line.rsplit(': ', 1)
            if name in robot_names:
                runtimes[name] = float(val)
    return runtimes


def _parse_align_runtimes(path: Path, robot_names: set) -> dict:
    """Parse align/runtime.txt, returning {pair_name: seconds} for pairs where both robots are in robot_names.

    Robot names contain underscores, so every possible split point is tried to identify valid (r1, r2) pairs.
    Includes self-pairs (r1 == r2) and cross-pairs.
    """
    runtimes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ': ' not in line:
                continue
            name, val = line.rsplit(': ', 1)
            for i in range(1, len(name)):
                if name[i] == '_':
                    r1, r2 = name[:i], name[i+1:]
                    if r1 in robot_names and r2 in robot_names:
                        runtimes[name] = float(val)
                        break
    return runtimes


def _parse_rpgo_runtime(path: Path) -> float:
    """Parse offline_rpgo/runtime.txt, returning the runtime in seconds as a single float."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return float(lines[-1])


def main():

    # Set experiment configuration
    user = getpass.getuser()
    dataset_folder = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset'
    #robot_names_text = ["acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek"]
    robot_names_text = ["acl_jackal", "acl_jackal2"]
    dataset_number = "1014"
    repository_with_results = "ROMAN_DEVEL"
    sequence_name = "Kimera-Multi_MG/Hard/" + DATASET_SEQUENCE[dataset_number] + "/" + "_".join(robot_names_text)

    # Load the estimated data
    est_data_dir = Path('/home/') / user / 'Research' / repository_with_results / 'results' / sequence_name
    est_data_lst: list[OdometryData] = []
    for rn in robot_names_text:
        est_data = OdometryData.from_csv(est_data_dir / 'offline_rpgo' / (rn + ".csv"), "map", 'robot', 
                                CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        est_data_lst.append(est_data)

    # Load the ground truth data
    gt_data_lst: list[OdometryData] = []
    for rn in robot_names_text:
        gt_data = OdometryData.from_csv(dataset_folder / 'data' / 'ground_truth' / dataset_number / (rn + '_gt_odom.csv'), 
                                        "world", "robot", CoordinateFrame.FLU, True, None, ts_in_ns=True)
        gt_data_lst.append(gt_data)

    # Match time spans before any evaluation (mirrors ROMAN_DEVEL evaluate behavior)
    est_data_lst, gt_data_lst = PathData.make_start_and_end_times_match(est_data_lst, gt_data_lst)

    # Calculate individual RMS ATE, among other metrics
    for i in range(len(est_data_lst)):
        print("=========== Individual Trajectory", robot_names_text[i], "for dataset: Kimera-Multi ============")
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data_lst[i], est_data_lst[i],
                                                                                max_diff=0.1, visualize=False)
        print("RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        print("RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse)
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)

    # Calculate merged RMS ATE:
    if len(est_data_lst) > 1:
        est_data: OdometryData = PathData.concatenate_PathData(est_data_lst).to_OdometryData('odom', 'base_link')
        gt_data: PathData = PathData.concatenate_PathData(gt_data_lst)

        # Calculate RMS ATE, among other metrics
        print("\n========== Merged Trajectories for dataset: Kimera-Multi ==========")
        metrics_dictionary, est_data_align, gt_data_align = OdometryData.align_and_calculate_traj_errors(gt_data, 
                                                                        est_data, max_diff=0.1, visualize=True)
        print("RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        print("RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)

        print("RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse)
        print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)

    # Load and print the runtimes
    robot_names_set = set(robot_names_text)
    map_runtimes = _parse_map_runtimes(est_data_dir / 'map' / 'runtime.txt', robot_names_set)
    align_runtimes = _parse_align_runtimes(est_data_dir / 'align' / 'runtime.txt', robot_names_set)
    rpgo_runtime = _parse_rpgo_runtime(est_data_dir / 'offline_rpgo' / 'runtime.txt')

    print("\n========== Map Runtimes ==========")
    for rn, t in map_runtimes.items():
        print(f"  {rn}: {t / 60:.3f} min")
    map_total = sum(map_runtimes.values())
    print(f"  Total: {map_total / 60:.3f} min ({map_total / 3600:.3f} hr)")

    print("\n========== Align Runtimes ==========")
    for pair, t in align_runtimes.items():
        print(f"  {pair}: {t / 60:.3f} min")
    align_total = sum(align_runtimes.values())
    print(f"  Total: {align_total / 60:.3f} min ({align_total / 3600:.3f} hr)")

    print("\n========== Offline RPGO Runtime ==========")
    print(f"  Total: {rpgo_runtime / 60:.3f} min ({rpgo_runtime / 3600:.3f} hr)")

    total_runtime = map_total + align_total + rpgo_runtime
    print(f"\n========== Combined Total Runtime ==========")
    print(f"  Total: {total_runtime / 60:.3f} min ({total_runtime / 3600:.3f} hr)")

if __name__ == "__main__":
    main()
import getpass
import re
from robotdataprocess import OdometryData, CoordinateFrame
from typing import List

def _pair_label(name_a: str, name_b: str) -> str:
    def abbrev(n):
        m = re.match(r'([A-Za-z]+)(\d+)', n)
        return (m.group(1)[0].upper() + m.group(2)) if m else n
    return abbrev(name_a) + abbrev(name_b)

def load_gt_data_ROMAN(dataset_name: str, robot_names: List) -> List[OdometryData]:
    """
    Load ground truth trajectories for a set of robots from poseGT.csv.

    Returns:
        List of OdometryData in the same order as robot_names, in FLU frame.
    """
    user = getpass.getuser()
    return [
        OdometryData.from_csv(
            '/media/' + user + '/T73/Hercules_datasets/' + dataset_name +
            '/extract/files_for_roman_baseline/' + rn + '/poseGT.csv',
            'world', 'robot', CoordinateFrame.FLU, True, None)
        for rn in robot_names
    ]

def load_est_data_ROMAN(dataset_name: str, method: str, robot_names: List) -> List[OdometryData]:
    """
    Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    user = getpass.getuser()
    run_name = "_".join(robot_names)
    return [
        OdometryData.from_csv(
            '/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method +
            '/' + run_name + '/offline_rpgo/' + rn + '.csv',
            "map", 'robot' + str(i), CoordinateFrame.FLU, True, [0, 1, 2, 3, 4, 5, 6, 7], ts_in_ns=True, reorder_data=False)
        for i, rn in enumerate(robot_names)
    ]

def load_kimera_rpgo_first_stage_est_data_ROMAN(dataset_name: str, method: str, robot_names: List) -> List[OdometryData]:
    """
    Load estimated trajectories for a set of robots from ROMAN offline RPGO output.

    Returns:
        List of OdometryData in the same order as robot_names.
    """
    user = getpass.getuser()
    run_name = "_".join(robot_names)
    result_dir = '/home/' + user + '/Research/ROMAN_DEVEL/results/hercules_' + dataset_name + '_' + method + '/' + \
                  run_name + '/offline_rpgo/'
    names_override = {chr(97 + i): name for i, name in enumerate(robot_names)}
    return [
        OdometryData.from_g2o(result_dir + 'pre_optimize/result.g2o', result_dir + 'dense/odom_all.time.txt', rn,
            "map", 'robot' + str(i), CoordinateFrame.FLU, names_override)
        for i, rn in enumerate(robot_names)
    ]
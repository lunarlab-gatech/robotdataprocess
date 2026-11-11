import argparse
import copy
from dataclasses import dataclass
from decimal import Decimal
import functools
import json
from pathlib import Path
from robotdataprocess import CoordinateFrame, OdometryData, PathData
from robotdataprocess.data_types.SLAMData import SLAMData
from robotdataprocess.eval.RobotGroup import RobotGroupViz
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator
from robotdataprocess.eval.SLAMMethod import SLAMMethod
from typing import Dict, List

# The Vicon PC's UTC offset on 2026-11-06 (US Eastern Standard Time).
VICON_UTC_OFFSET_HOURS: float = -5.0

# Where the IFL Vicon exports and the robots' bags live (the container's ~/data/bags).
BAGS_DIR: Path = Path.home() / 'data' / 'bags'
IFL_DIR: Path = BAGS_DIR / 'IFL'

# The method evaluated and each robot's plot color.
METHOD: SLAMMethod = SLAMMethod('MG_TS_SM', 'MeronomyGraph', 'MG_TS_SM', '#ff7f0e')
ROBOT_COLORS: Dict[str, str] = {'Go2': '#FF0101', 'Thunderstrike': '#0014FF'}


@dataclass(frozen=True)
class IFLRun:
    """
    One robot's IFL run: its Vicon capture and the bag recorded alongside it.

    Attributes:
        vicon_stem: Vicon capture file name without extension (shared by its .xlsx and .xcp).
        vicon_object: The robot's Vicon subject name.
        bag_name: The run's bag folder under ``BAGS_DIR``.
        odom_topic: The bag's odometry topic, stamped on the same clock as the SLAM estimate.
    """
    vicon_stem: str
    vicon_object: str
    bag_name: str
    odom_topic: str


IFL_RUNS: Dict[str, IFLRun] = {
    'Go2': IFLRun('Rosbag_Recording_UniTreeGo2_Thunderstrike_SparseObjects', 'Lunar_UniTreeGo2',
                  'Go2_2026-11-06_123618', '/Go2/odom'),
    'Thunderstrike': IFLRun('Rosbag_Recording_UniTreeGo2_Thunderstrike_SparseObjects 2', 'Lunar_ThunderStrike',
                            'Thunderstrike_2026-11-06_125037', '/Thunderstrike/odom'),
}


@functools.lru_cache(maxsize=None)
def _load_robot_gt(robot: str) -> OdometryData:
    """
    Loads one robot's Vicon ground truth, moved onto the robot's own clock.

    The robot's estimate is stamped on its own clock, while Vicon stamps the Vicon PC's wall
    clock, so the offset is estimated against the run's bag odometry (same clock as the estimate).
    The Vicon object is taken to be the robot's body frame.

    Args:
        robot: Robot name, a key of ``IFL_RUNS``.
    Returns:
        OdometryData: The ground truth in the Vicon world frame, on the robot's clock.
    """
    run: IFLRun = IFL_RUNS[robot]
    start: Decimal = PathData.vicon_capture_start_time(IFL_DIR / f'{run.vicon_stem}.xcp', VICON_UTC_OFFSET_HOURS)
    gt = OdometryData.from_vicon_xlsx(IFL_DIR / f'{run.vicon_stem}.xlsx', 'vicon', robot, CoordinateFrame.NONE,
                                      start, run.vicon_object)
    odom = OdometryData.from_ros2_bag(BAGS_DIR / run.bag_name, run.odom_topic, CoordinateFrame.NONE)
    offset: Decimal = PathData.estimate_time_offset(odom, gt)
    print(f"{robot}: Vicon -> robot clock offset {offset} s")
    gt.shift_timestamps(offset)
    return gt


def load_gt_data(dataset_seq: str, robot_names: List[str]) -> List[OdometryData]:
    """
    Loads the IFL Vicon ground truth for each robot, as ``SLAMEvaluator``'s ``load_gt_data_fn``.

    Args:
        dataset_seq: The dataset sequence (always ``"live"`` for the Go2 dataset; unused).
        robot_names: Robots to load, each a key of ``IFL_RUNS``.
    Returns:
        List[OdometryData]: Ground truth in ``robot_names`` order, copied so callers may modify it.
    """
    return [copy.deepcopy(_load_robot_gt(robot)) for robot in robot_names]


def evaluate_each_robot(mg_root: Path, critical_invocation_params: Dict, save_dir: Path) -> None:
    """
    Aligns each robot's estimate to its own ground truth (one fit per robot, unlike the shared fit
    of ``ate_split_table``), printing its RMS and max ATE and saving its 2D trajectory plot.

    Args:
        mg_root: MeronomyGraph checkout with its results/.
        critical_invocation_params: The run's data-affecting args, to find its hash-addressed results.
        save_dir: Where the ``traj_<robot>_self_aligned.pdf`` plots are saved.
    """
    robots: List[str] = sorted(IFL_RUNS)
    system_params = SLAMData.load_system_params(mg_root, 'go2', 'live', METHOD.base_param_config)
    estimates: List[OdometryData] = SLAMData.load_est_data(mg_root, system_params, robots, critical_invocation_params)
    save_dir.mkdir(parents=True, exist_ok=True)
    for robot, est, gt in zip(robots, estimates, load_gt_data('live', robots)):
        est_align, gt_align, _, _ = SLAMEvaluator.align_merged_trajectories([robot], [est], [gt])
        ate = PathData.calculate_traj_errors(gt_align, est_align).APE.translation_part
        print(f"{robot}: self-aligned RMS ATE {ate.rmse:.3f} m, max {ate.max:.3f} m")
        PathData.visualize_2D([est_align, gt_align], [False, True], [ROBOT_COLORS[robot]] * 2, [robot] * 2,
                              save_path=str(save_dir / f'traj_{robot}_self_aligned.pdf'), show_grid=True)


def main():
    """
    Evaluates a Go2 + Thunderstrike MeronomyGraph run against the IFL Vicon ground truth.

    Results are looked up in the MeronomyGraph hash-addressed result tree with the run folder's
    ``critical_invocation_params.json``; see :meth:`SLAMEvaluator.run_evaluation` for the outputs,
    plus :func:`evaluate_each_robot`'s per-robot ATE and plots under ``self_aligned/``.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.strip().splitlines()[0])
    parser.add_argument('mg_root', type=Path, help='MeronomyGraph checkout with its results/')
    parser.add_argument('run_dir', type=Path, help='Run folder holding critical_invocation_params.json')
    parser.add_argument('--figures-dir', type=Path, default=Path.home() / 'data' / 'figures',
                        help='Where the figures and tables are written')
    args = parser.parse_args()

    critical_invocation_params: Dict = json.loads((args.run_dir / 'critical_invocation_params.json').read_text())
    viz_config = RobotGroupViz(name_map=None, robot_name_to_color=ROBOT_COLORS,
                               image_path=None, image_x_edge=None, image_extent_offsets=None)
    robot_groups = SLAMEvaluator.make_robot_groups('go2', 'live', [('Go2', 'Thunderstrike')], viz_config)
    output_dir = Path('IFL') / args.run_dir.name
    SLAMEvaluator.run_evaluation(args.mg_root, output_dir, [METHOD], robot_groups,
                                 critical_invocation_params, args.figures_dir, load_gt_data,
                                 failure_extent_fraction=0.10)
    evaluate_each_robot(args.mg_root, critical_invocation_params, args.figures_dir / output_dir / 'self_aligned')


if __name__ == '__main__':
    main()

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame, OdometryData
from robotdataprocess.data_types.SLAMData import SLAMData
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator
from scipy.spatial.transform import Rotation as R
import shutil
import tempfile
from typing import List
import unittest


class _FakeOfflineRPGOParams:
    def __init__(self, sparsified: bool):
        self.sparsified = sparsified


class _FakeSystemParams:
    """Stand-in for SystemParams that resolves result directories by a plain, readable
    layout rather than the real hash-addressed reconstruction.

    Mirrors the real SystemParams: ``dataset_name``/``dataset_version`` are stored on
    construction (as attributes of the same name), so ``rpgo_result_dir``/``align_result_dir``/
    ``mapping_result_dir`` no longer take them as call arguments -- callers read them off the
    instance instead. Every directory lives under ``<results_root>/<dataset_version>/<method>/``:
    ``rpgo/<robot_a>_<robot_b>``, ``align/<robot_a>_<robot_b>``, and ``mapping/<robot>``. Like the
    real thing, neither ``rpgo_result_dir`` nor ``align_result_dir`` is order-invariant, so both
    assert that their robot names arrive already sorted -- an unsorted caller would silently
    address a different directory in production.

    Attributes:
        dataset_name: Read directly by ``SLAMEvaluator.save_merged_ate_figures`` for its output path.
        dataset_version: Read directly by ``SLAMEvaluator.calculate_merged_ate``/
            ``save_merged_ate_figures`` (passed to ``load_gt_data_fn`` as ``dataset_seq``), and
            used here for the results directory layout.
        offline_rpgo_params: Holds ``sparsified``, read by ``SLAMEvaluator.load_LC_data``.
    """

    dataset_name: str
    dataset_version: str
    offline_rpgo_params: _FakeOfflineRPGOParams

    def __init__(self, dataset_name: str, dataset_version: str, method: str, sparsified: bool):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.offline_rpgo_params = _FakeOfflineRPGOParams(sparsified)
        self.method: str = method

    def _method_dir(self, results_root) -> Path:
        return Path(results_root) / self.dataset_version / self.method

    def rpgo_result_dir(self, results_root, sorted_robot_names, critical_invocation_params) -> Path:
        assert list(sorted_robot_names) == sorted(sorted_robot_names), \
            f"rpgo_result_dir requires sorted robot names, got {list(sorted_robot_names)}"
        return self._method_dir(results_root) / 'rpgo' / '_'.join(sorted_robot_names)

    def align_result_dir(self, results_root, name_a, name_b, critical_invocation_params) -> Path:
        assert name_a <= name_b, f"align_result_dir requires sorted robot names, got ({name_a}, {name_b})"
        return self._method_dir(results_root) / 'align' / f'{name_a}_{name_b}'

    def mapping_result_dir(self, results_root, robot_name, critical_invocation_params) -> Path:
        return self._method_dir(results_root) / 'mapping' / robot_name

@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestCalculateMergedAte(unittest.TestCase):
    """Regression tests for SLAMEvaluator.calculate_merged_ate, pinned to its current
    output on a small deterministic fixture (two synthetic robot trajectories with
    non-rigid drift + yaw error, so ATE/RPE don't collapse to zero after alignment).
    Expected values were captured from one real run of the function against
    ``tests/files/test_SLAMEvaluator/calculate_merged_ate`` and hardcoded below --
    a future change to the alignment/error-metric pipeline that shifts these numbers
    should fail here even if it doesn't fail anywhere else.
    """

    MG_ROOT = Path(__file__).parent / 'files' / 'test_SLAMEvaluator' / 'calculate_merged_ate'
    DATASET_SEQ = 'test_dataset_seq'
    METHOD = 'ROMAN'

    @staticmethod
    def _make_gt_odometry(x0: float, y0: float, y_amp: float, y_freq: float) -> OdometryData:
        """Rebuilds the noise-free ground-truth trajectory used to generate the fixture
        estimates -- independent of calculate_merged_ate itself, just the same simple
        parametric curve (straight-line x, sinusoidal y, no yaw) used to construct it."""
        t = np.arange(5, dtype=float)
        positions = np.stack([x0 + t, y0 + y_amp * np.sin(y_freq * t), np.zeros(5)], axis=1)
        orientations = R.from_euler('z', np.zeros(5)).as_quat()
        return OdometryData('map', 'robot', t, positions, orientations, CoordinateFrame.NONE)

    @classmethod
    def _load_gt_data_fn(cls, dataset_seq: str, robot_names):
        gt_by_name = {
            'robotA': cls._make_gt_odometry(0.0, 0.0, 0.02, 1.0),
            'robotB': cls._make_gt_odometry(0.0, 5.0, -0.02, 1.0),
        }
        return [gt_by_name[rn] for rn in robot_names]

    @classmethod
    def _build_slam_data(cls, robot_names: List[str]) -> SLAMData:
        """Builds a SLAMData straight from the fixture's loaders, bypassing
        from_MeronomyGraph -- which would also try to load timing/data-size files this
        fixture doesn't have."""
        system_params = _FakeSystemParams(dataset_name='kimera_multi', dataset_version=cls.DATASET_SEQ,
                                          method=cls.METHOD, sparsified=False)
        sorted_names = sorted(robot_names)
        estimated_trajectories = SLAMData.load_est_data(cls.MG_ROOT, system_params, sorted_names, {})
        try:
            pre_opt_est_trajectories = SLAMData.load_kimera_rpgo_first_stage_est_data(
                cls.MG_ROOT, system_params, sorted_names, {})
        except Exception:
            pre_opt_est_trajectories = []
        alignment_lc, inlier_lc = SLAMData.load_LC_data(cls.MG_ROOT, system_params, sorted_names, {})
        return SLAMData(system_params, sorted_names, estimated_trajectories, alignment_lc, inlier_lc,
                        timing={"align": {}, "mapping": {}, "offline_rpgo": {}}, data_size_mb=0.0,
                        pre_opt_est_trajectories=pre_opt_est_trajectories)

    def test_two_robot_group_pins_metrics(self):
        # calculate_merged_ate is purely computational -- save_merged_ate_figures is the
        # figure-saving counterpart, tested separately.
        slam_data = self._build_slam_data(['robotA', 'robotB'])
        result = SLAMEvaluator.calculate_merged_ate(slam_data, self._load_gt_data_fn, rpe_delta=1.0)

        self.assertAlmostEqual(result.first_stage_metrics.APE.translation_part.rmse, 0.08788216952120878, places=8)
        self.assertAlmostEqual(result.first_stage_metrics.APE.rotation_angle_deg.rmse, 7.348450557420504, places=6)
        self.assertAlmostEqual(result.first_stage_metrics.RPE.translation_part.rmse, 0.46946884541432593, places=8)

        self.assertAlmostEqual(result.merged_metrics.APE.translation_part.rmse, 0.014142135626144945, places=8)
        self.assertAlmostEqual(result.merged_metrics.APE.rotation_angle_deg.rmse, 2.4494692640072837, places=6)
        self.assertAlmostEqual(result.merged_metrics.RPE.translation_part.rmse, 0.15171134070137318, places=8)
        self.assertAlmostEqual(result.merged_metrics.RPE.rotation_angle_deg.rmse, 1.6955531119021618, places=6)

        self.assertEqual(len(result.robot_metrics), 2)
        self.assertAlmostEqual(result.robot_metrics[0].APE.translation_part.rmse, 0.01414213562614496, places=8)
        self.assertAlmostEqual(result.robot_metrics[0].APE.rotation_angle_deg.rmse, 2.449469264007282, places=6)
        self.assertAlmostEqual(result.robot_metrics[1].APE.translation_part.rmse, 0.014142135626144931, places=8)
        self.assertAlmostEqual(result.robot_metrics[1].APE.rotation_angle_deg.rmse, 2.4494692640072846, places=6)

    def test_single_robot_self_alignment_with_missing_first_stage_file(self):
        def load_gt_data_fn_single(dataset_seq, robot_names):
            return [self._make_gt_odometry(0.0, 0.0, 0.02, 1.0)]

        slam_data = self._build_slam_data(['robotA'])
        result = SLAMEvaluator.calculate_merged_ate(slam_data, load_gt_data_fn_single, rpe_delta=1.0)

        # No pre_optimize/ directory exists for the single-robot fixture, so the
        # first-stage load raises and is caught -- exercising that except branch.
        self.assertIsNone(result.first_stage_metrics)

        # A single-robot group is a self-alignment case: the "merged" trajectory is
        # just that robot's own trajectory, so its per-robot metrics must equal the
        # merged metrics exactly (computed via a different code path: seperate_PathData
        # splitting the merged alignment back apart, vs. the merge/align call itself).
        self.assertEqual(len(result.robot_metrics), 1)
        self.assertEqual(result.robot_metrics[0].APE.translation_part.rmse, result.merged_metrics.APE.translation_part.rmse)

        self.assertAlmostEqual(result.merged_metrics.APE.translation_part.rmse, 0.03229292806125464, places=8)
        self.assertAlmostEqual(result.merged_metrics.APE.rotation_angle_deg.rmse, 5.041425839526853, places=6)
        self.assertAlmostEqual(result.merged_metrics.RPE.translation_part.rmse, 0.07602451076923582, places=8)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestVisualizeMergedAte(TestCalculateMergedAte):
    """Tests SLAMEvaluator.save_merged_ate_figures against the same fixture as
    TestCalculateMergedAte -- reuses its GT/system_params setup, since both methods load the
    same estimated/ground-truth data and only differ in what they do with it (metrics vs. figures).
    """

    def setUp(self):
        self.figures_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.figures_dir, ignore_errors=True)

    def test_two_robot_group_saves_expected_figures(self):
        viz_config = {
            "image_path": None, "x_edge": None,
            "robot_name_to_color": {"robotA": "#FFA501", "robotB": "#0014FF"},
        }
        slam_data = self._build_slam_data(['robotA', 'robotB'])
        SLAMEvaluator.save_merged_ate_figures(
            slam_data, self.METHOD, self._load_gt_data_fn, self.figures_dir, viz_config)

        expected_files = [
            'kimera_multi/test_dataset_seq/traj/traj_RA-RB_ROMAN.pdf',
            'kimera_multi/test_dataset_seq/traj/traj_RA-RB_ROMAN_onlyGT.pdf',
            'kimera_multi/test_dataset_seq/ALL/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
            'kimera_multi/test_dataset_seq/ONLY_INTER_LC/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
            'kimera_multi/test_dataset_seq/ONLY_INTRA_LC/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
        ]
        for rel_path in expected_files:
            path = self.figures_dir / rel_path
            self.assertTrue(path.is_file(), f"Expected figure not saved: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Figure saved but empty: {path}")

    def test_single_robot_group_saves_expected_figures(self):
        # A single-robot group has no inter-robot pairs at all, so load_LC_data only needs an
        # empty intra-robot inlier file here (no inter-robot inlier file, unlike the 2-robot case).
        viz_config = {
            "image_path": None, "x_edge": None,
            "robot_name_to_color": {"robotA": "#FFA501"},
        }
        slam_data = self._build_slam_data(['robotA'])
        SLAMEvaluator.save_merged_ate_figures(
            slam_data, self.METHOD,
            lambda dataset_seq, robot_names: [self._make_gt_odometry(0.0, 0.0, 0.02, 1.0)],
            self.figures_dir, viz_config)

        expected_files = [
            'kimera_multi/test_dataset_seq/traj/traj_RA_ROMAN.pdf',
            'kimera_multi/test_dataset_seq/traj/traj_RA_ROMAN_onlyGT.pdf',
        ]
        for rel_path in expected_files:
            path = self.figures_dir / rel_path
            self.assertTrue(path.is_file(), f"Expected figure not saved: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Figure saved but empty: {path}")


if __name__ == '__main__':
    unittest.main()

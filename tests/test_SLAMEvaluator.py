from decimal import Decimal
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame, LoopClosureFilterMode, OdometryData
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator
from scipy.spatial.transform import Rotation as R
import shutil
import tempfile
from typing import Dict, List, Sequence, Tuple
import unittest


class _FakeOfflineRPGOParams:
    def __init__(self, sparsified: bool):
        self.sparsified = sparsified


class _FakeSystemParams:
    """Stand-in for SystemParams that resolves result directories by a plain, readable
    layout rather than the real hash-addressed reconstruction.

    Every directory lives under ``<results_root>/<dataset_name>/<method>/``, mirroring
    how the real SystemParams gives each (dataset, method) its own result tree:
    ``rpgo/<robot_a>_<robot_b>``, ``align/<robot_a>_<robot_b>``, and ``mapping/<robot>``.
    Like the real thing, neither ``rpgo_result_dir`` nor ``align_result_dir`` is
    order-invariant, so both assert that their robot names arrive already sorted --
    an unsorted caller would silently address a different directory in production.

    Attributes:
        offline_rpgo_params: Holds ``sparsified``, read by ``SLAMEvaluator.load_LC_data``.
    """

    offline_rpgo_params: _FakeOfflineRPGOParams

    def __init__(self, method: str, sparsified: bool):
        self.offline_rpgo_params = _FakeOfflineRPGOParams(sparsified)
        self.method: str = method

    def _method_dir(self, results_root, dataset_name: str) -> Path:
        return Path(results_root) / dataset_name / self.method

    def rpgo_result_dir(self, results_root, dataset_prefix, dataset_name, sorted_robot_names,
                        critical_invocation_params) -> Path:
        assert list(sorted_robot_names) == sorted(sorted_robot_names), \
            f"rpgo_result_dir requires sorted robot names, got {list(sorted_robot_names)}"
        return self._method_dir(results_root, dataset_name) / 'rpgo' / '_'.join(sorted_robot_names)

    def align_result_dir(self, results_root, dataset_prefix, dataset_name, name_a, name_b,
                         critical_invocation_params) -> Path:
        assert name_a <= name_b, f"align_result_dir requires sorted robot names, got ({name_a}, {name_b})"
        return self._method_dir(results_root, dataset_name) / 'align' / f'{name_a}_{name_b}'

    def mapping_result_dir(self, results_root, dataset_prefix, dataset_name, robot_name,
                           critical_invocation_params) -> Path:
        return self._method_dir(results_root, dataset_name) / 'mapping' / robot_name


def _expected_inlier_lc(rpgo_dir: Path, sorted_names, time_subdir: str) -> LoopClosureData:
    """Reproduces SLAMEvaluator.load_LC_data's inlier-assembly loop against a chosen time
    subdirectory ('sparse' or 'dense'), independent of whatever load_LC_data itself
    picks -- this is the "known correct" answer the loader's output is checked against.
    """
    letter_by_name = {name: chr(97 + i) for i, name in enumerate(sorted_names)}
    names_override = {chr(97 + i): name for i, name in enumerate(sorted_names)}
    time_path = rpgo_dir / time_subdir / 'odom_all.time.txt'

    lc_list = []
    for name_a, name_b in itertools.combinations_with_replacement(sorted_names, 2):
        letter_a, letter_b = letter_by_name[name_a], letter_by_name[name_b]
        g2o_filename = f'inlier_lc_intra_{letter_a}.g2o' if name_a == name_b \
            else f'inlier_lc_inter_{letter_a}_{letter_b}.g2o'
        lc_list.append(LoopClosureData.from_g2o(rpgo_dir / g2o_filename, time_path,
                                                names_override=names_override))

    merged = LoopClosureData.merge(lc_list)
    merged.prune_duplicates()
    return merged


def _lc_keys(lc: LoopClosureData):
    return set(zip(lc.names, lc.timestamps_a, lc.timestamps_b))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadLCDataROMANSparsified(unittest.TestCase):
    """Regression test for SLAMEvaluator.load_LC_data's inlier LC loading with sparsified=True
    (Kimera-Multi's offline RPGO config). The inlier g2o files Kimera-RPGO writes carry
    vertex keys in the *sparse* keyframe indexing (final_g2o_file is built from
    odom_sparse_all_g2o_file when sparsified), so they must be paired with
    sparse/odom_all.time.txt, not dense/odom_all.time.txt.
    """

    ROMAN_ROOT = Path(__file__).parent / 'files' / 'test_ROMAN' / 'sparsified'
    DATASET_NAME = 'campus_outdoor_1014_compressed'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['acl_jackal', 'acl_jackal2']
    RPGO_DIR = ROMAN_ROOT / 'results' / DATASET_NAME / METHOD / 'rpgo' / 'acl_jackal_acl_jackal2'

    def test_inlier_lc_uses_sparse_timestamps(self):
        evaluator = SLAMEvaluator(self.ROMAN_ROOT)
        system_params = _FakeSystemParams(self.METHOD, sparsified=True)
        _, lc_inlier = evaluator.load_LC_data(
            system_params, 'kimera_multi', self.DATASET_NAME,
            self.ROBOT_NAMES, {}, lc_filter=LoopClosureFilterMode.ALL)

        expected = _expected_inlier_lc(self.RPGO_DIR, sorted(self.ROBOT_NAMES), 'sparse')
        self.assertEqual(lc_inlier.num_loop_closures, expected.num_loop_closures)
        self.assertEqual(_lc_keys(lc_inlier), _lc_keys(expected))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadLCDataROMANNonSparsified(unittest.TestCase):
    """Regression test for SLAMEvaluator.load_LC_data's inlier LC loading with sparsified=False
    (Hercules/AirMuseum's offline RPGO config). The inlier g2o files carry vertex keys in
    the dense keyframe indexing (final_g2o_file is dense_g2o_file itself when not
    sparsified), matching dense/odom_all.time.txt.
    """

    ROMAN_ROOT = Path(__file__).parent / 'files' / 'test_ROMAN' / 'non_sparsified'
    DATASET_NAME = 'V2.4.F'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['Drone2', 'Husky1']
    RPGO_DIR = ROMAN_ROOT / 'results' / DATASET_NAME / METHOD / 'rpgo' / 'Drone2_Husky1'

    def test_inlier_lc_uses_dense_timestamps(self):
        evaluator = SLAMEvaluator(self.ROMAN_ROOT)
        system_params = _FakeSystemParams(self.METHOD, sparsified=False)
        _, lc_inlier = evaluator.load_LC_data(
            system_params, 'hercules', self.DATASET_NAME,
            self.ROBOT_NAMES, {}, lc_filter=LoopClosureFilterMode.ALL)

        expected = _expected_inlier_lc(self.RPGO_DIR, sorted(self.ROBOT_NAMES), 'dense')
        self.assertEqual(lc_inlier.num_loop_closures, expected.num_loop_closures)
        self.assertEqual(_lc_keys(lc_inlier), _lc_keys(expected))


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

    def setUp(self):
        self.evaluator = SLAMEvaluator(self.MG_ROOT)
        self.system_params = _FakeSystemParams(self.METHOD, sparsified=False)

    def test_two_robot_group_pins_metrics(self):
        # calculate_merged_ate is purely computational -- visualize_merged_ate is the
        # figure-saving counterpart, tested separately.
        result = self.evaluator.calculate_merged_ate(
            self.system_params, 'kimera_multi', self.DATASET_SEQ, self.METHOD, ['robotA', 'robotB'], {},
            self._load_gt_data_fn, rpe_delta=1.0)

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

        result = self.evaluator.calculate_merged_ate(
            self.system_params, 'kimera_multi', self.DATASET_SEQ, self.METHOD, ['robotA'], {},
            load_gt_data_fn_single, rpe_delta=1.0)

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
    """Tests SLAMEvaluator.visualize_merged_ate against the same fixture as
    TestCalculateMergedAte -- reuses its GT/system_params setup, since both methods load the
    same estimated/ground-truth data and only differ in what they do with it (metrics vs. figures).
    """

    def setUp(self):
        super().setUp()
        self.figures_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.figures_dir, ignore_errors=True)

    def test_two_robot_group_saves_expected_figures(self):
        viz_config = {
            "image_path": None, "x_edge": None,
            "robot_name_to_color": {"robotA": "#FFA501", "robotB": "#0014FF"},
        }
        self.evaluator.save_merged_ate_figures(
            self.system_params, 'kimera_multi', self.DATASET_SEQ, self.METHOD, ['robotA', 'robotB'], {},
            self._load_gt_data_fn, self.figures_dir, viz_config)

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
        self.evaluator.save_merged_ate_figures(
            self.system_params, 'kimera_multi', self.DATASET_SEQ, self.METHOD, ['robotA'], {},
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

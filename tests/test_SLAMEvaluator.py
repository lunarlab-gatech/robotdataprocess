import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from operator import attrgetter
import os
import pandas as pd
from pathlib import Path
from robotdataprocess import CoordinateFrame, LoopClosureFilterMode, OdometryData, PathData, TableData
from robotdataprocess.data_types.SLAMData import SLAMData
from robotdataprocess.eval.RobotGroup import RobotGroup, RobotGroupViz
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator
from robotdataprocess.eval.SLAMEvaluatorResult import SLAMResult
from robotdataprocess.eval.SLAMMethod import SLAMMethod
from scipy.spatial.transform import Rotation as R
import shutil
import tempfile
from typing import List, Optional
import unittest
import unittest.mock


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
        dataset_name: Used here for the results directory layout.
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
                        data_num_objects_sent=None,
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

        # Aligned trajectories are stored per robot, in robot name order, with est/GT time-matched
        self.assertEqual(len(result.est_align_list), 2)
        self.assertEqual(len(result.gt_align_list), 2)
        for est, gt in zip(result.est_align_list, result.gt_align_list):
            np.testing.assert_array_equal(est.timestamps.astype(float), gt.timestamps.astype(float))

        # GT is the noise-free fixture curve (x = t, y = +/-0.02 sin t about 0 or 5), robotA then robotB
        gt_y_A = [0.0, 0.016829, 0.018186, 0.002822, -0.015136]
        np.testing.assert_allclose(result.gt_align_list[0].positions[:, 0].astype(float), [0, 1, 2, 3, 4], atol=1e-6)
        np.testing.assert_allclose(result.gt_align_list[0].positions[:, 1].astype(float), gt_y_A, atol=1e-6)
        np.testing.assert_allclose(result.gt_align_list[1].positions[:, 1].astype(float), [5.0 - y for y in gt_y_A], atol=1e-6)

        # Estimates are aligned: the unaligned 1.01x drift ([0, 1.01, ...]) shifted by -0.02 in x, unlike GT or raw
        aligned_x = [-0.02, 0.99, 2.0, 3.01, 4.02]
        np.testing.assert_allclose(result.est_align_list[0].positions[:, 0].astype(float), aligned_x, atol=1e-6)
        np.testing.assert_allclose(result.est_align_list[1].positions[:, 0].astype(float), aligned_x, atol=1e-6)
        np.testing.assert_allclose(result.est_align_list[1].positions[:, 1].astype(float), [5.0 - y for y in gt_y_A], atol=1e-6)

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
        viz_config = RobotGroupViz(
            name_map=None, robot_name_to_color={"robotA": "#FFA501", "robotB": "#0014FF"},
            image_path=None, image_x_edge=None, image_extent_offsets=None,
        )
        slam_data = self._build_slam_data(['robotA', 'robotB'])
        result = SLAMEvaluator.calculate_merged_ate(slam_data, self._load_gt_data_fn, rpe_delta=1.0)
        SLAMEvaluator.save_merged_ate_figures(
            slam_data, result, self.METHOD, 'RA-RB', self._load_gt_data_fn, self.figures_dir, viz_config)

        expected_files = [
            'traj/est_and_gt/individual_methods/traj_RA-RB_ROMAN.pdf',
            'traj/gt/traj_RA-RB_ROMAN_onlyGT.pdf',
            'ALL/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
            'ONLY_INTER_LC/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
            'ONLY_INTRA_LC/traj_lc/traj_lc_RA-RB_ROMAN.pdf',
        ]
        for rel_path in expected_files:
            path = self.figures_dir / rel_path
            self.assertTrue(path.is_file(), f"Expected figure not saved: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Figure saved but empty: {path}")

    def test_single_robot_group_saves_expected_figures(self):
        # A single-robot group has no inter-robot pairs at all, so load_LC_data only needs an
        # empty intra-robot inlier file here (no inter-robot inlier file, unlike the 2-robot case).
        viz_config = RobotGroupViz(
            name_map=None, robot_name_to_color={"robotA": "#FFA501"},
            image_path=None, image_x_edge=None, image_extent_offsets=None,
        )
        slam_data = self._build_slam_data(['robotA'])
        load_gt_data_fn_single = lambda dataset_seq, robot_names: [self._make_gt_odometry(0.0, 0.0, 0.02, 1.0)]
        result = SLAMEvaluator.calculate_merged_ate(slam_data, load_gt_data_fn_single, rpe_delta=1.0)
        SLAMEvaluator.save_merged_ate_figures(
            slam_data, result, self.METHOD, 'RA', load_gt_data_fn_single, self.figures_dir, viz_config)

        expected_files = [
            'traj/est_and_gt/individual_methods/traj_RA_ROMAN.pdf',
            'traj/gt/traj_RA_ROMAN_onlyGT.pdf',
        ]
        for rel_path in expected_files:
            path = self.figures_dir / rel_path
            self.assertTrue(path.is_file(), f"Expected figure not saved: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Figure saved but empty: {path}")


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveAllMethodsTrajFigure(unittest.TestCase):
    """Tests SLAMEvaluator._save_all_methods_traj_figure by keeping its figure open (``plt.close``
    patched out) and inspecting the drawn lines and legend directly."""

    # Drawn colors are the input hue at visualize_2D's lightness 9/19 (est) or 3/19 (GT)
    RED_EST = (18 / 19, 0.0, 0.0)
    BLUE_EST = (0.0, 0.0, 18 / 19)
    GT_GRAY = (3 / 19, 3 / 19, 3 / 19)

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.viz_config = RobotGroupViz(name_map=None, robot_name_to_color={}, image_path=None,
                                        image_x_edge=None, image_extent_offsets=None)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _make_path(y: float, num_poses: int = 2) -> PathData:
        """Straight line along x at height y, so each trajectory is identifiable by its y value."""
        return PathData('map', np.arange(num_poses, dtype=float).astype(object),
                        np.array([[float(x), y, 0.0] for x in range(num_poses)], dtype=object),
                        np.array([[0, 0, 0, 1]] * num_poses, dtype=object), CoordinateFrame.NONE)

    def _make_result(self, ys: List[float]) -> SLAMResult:
        # Stored gt_align_list is 2 poses long, unlike the 3-pose full GT, to tell them apart
        return SLAMResult(None, None, [], [self._make_path(y) for y in ys], [self._make_path(-9.0) for _ in ys])

    def _save_and_get_figure(self, run_names, results_by_run, gt_data_lst, run_to_color, run_display_names, save_dir):
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.plt.close') as mock_close:
            SLAMEvaluator._save_all_methods_traj_figure(run_names, results_by_run, gt_data_lst, 'G\nRA-RB',
                                        self.viz_config, run_to_color, run_display_names, save_dir)
        fig = mock_close.call_args[0][0]
        self.addCleanup(plt.close, fig)
        return fig.axes[0]

    def test_two_runs_two_robots(self):
        # RUN_C is in results_by_run but not run_names, so must not be drawn
        results_by_run = {'RUN_A': self._make_result([0.0, 1.0]), 'RUN_B': self._make_result([2.0, 3.0]),
                          'RUN_C': self._make_result([7.0, 8.0])}
        gt_data_lst = [self._make_path(-1.0, 3), self._make_path(-2.0, 3)]
        save_dir = self.tmp_dir / 'not_yet_created'
        ax = self._save_and_get_figure(['RUN_A', 'RUN_B'], results_by_run, gt_data_lst,
                                       {'RUN_A': '#FF0000', 'RUN_B': '#0000FF', 'RUN_C': '#00FF00'},
                                       {'RUN_A': 'Run A Display'}, save_dir)

        # Saved into a freshly created directory under the sanitized label
        saved = save_dir / "traj_G_RA-RB.pdf"
        self.assertTrue(saved.is_file(), f"Expected figure not saved: {saved}")
        self.assertGreater(saved.stat().st_size, 0)

        # One legend entry per run (display name, falling back to run name), then GT
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["Run A Display", "RUN_B", "GT"])

        # Each drawn line, in order: (y value, color, unscaled dash pattern, num points)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 6)
        solid = (0, None)
        gt_dash = (0, (4, 1.65))
        expected = [(0.0, self.RED_EST, solid, 2), (1.0, self.RED_EST, solid, 2),
                    (2.0, self.BLUE_EST, solid, 2), (3.0, self.BLUE_EST, solid, 2),
                    (-1.0, self.GT_GRAY, gt_dash, 3), (-2.0, self.GT_GRAY, gt_dash, 3)]
        for line, (y, color, dash_pattern, num_points) in zip(lines, expected):
            ydata = np.asarray(line.get_ydata(), dtype=float)
            np.testing.assert_allclose(ydata, [y] * num_points)
            np.testing.assert_allclose(line.get_color(), color, atol=1e-9)
            self.assertEqual(line._unscaled_dash_pattern, dash_pattern)

    def test_single_run_single_robot(self):
        ax = self._save_and_get_figure(['X'], {'X': self._make_result([4.0])}, [self._make_path(-1.0, 3)],
                                       {'X': '#FF0000'}, {}, self.tmp_dir)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["X", "GT"])
        self.assertEqual(len(ax.get_lines()), 2)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveTimingTable(unittest.TestCase):
    """SLAMEvaluator._save_timing_table: every table, the alignment row included, uses each group's actual compute
    from get_timing_totals, with the total summing alignment, mapping, and RPGO."""

    SM_ALIGN: dict = {('ds', 's', 'a', 'b'): [1.0, 2.0], ('ds', 's', 'a', 'c'): [10.0, 20.0]}
    ROMAN_ALIGN: dict = {('ds', 's', 'a', 'b'): [1.0, 2.0], ('ds', 's', 'a', 'c'): [10.0, 20.0, 40.0]}

    def _group(self, align: dict) -> SLAMData:
        slam_data = object.__new__(SLAMData)
        slam_data.timing = {'align': align, 'mapping': {('ds', 's', 'a'): 100.0},
                            'offline_rpgo': {('ds', 's', 'a', 'b', 'c'): 1000.0}}
        return slam_data

    def test_alignment_and_total_are_actual_compute(self):
        tables = {}
        def capture(raw_df, title, **kwargs):
            tables[title] = raw_df
            return unittest.mock.MagicMock()  # stands in for the table object whose .to_latex is called
        slam_data_by_run = {'SM': {'g1': self._group(self.SM_ALIGN)}, 'ROMAN': {'g1': self._group(self.ROMAN_ALIGN)}}
        with unittest.mock.patch.object(SLAMEvaluator, 'make_highlighted_table', side_effect=capture), \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_timing_table(['SM', 'ROMAN'], ['g1'], {}, slam_data_by_run, Path(tmp_dir) / 'timing.pdf')

        align, total = tables["Alignment Runtime (s)"], tables["Total Runtime (s)"]
        self.assertEqual(align.loc['SM', 'g1'], 3.0 + 30.0)
        self.assertEqual(align.loc['ROMAN', 'g1'], 3.0 + 70.0)    # every attempt of every pair
        self.assertEqual(total.loc['SM', 'g1'], 33.0 + 100.0 + 1000.0)
        self.assertEqual(tables["Mapping Runtime (s)"].loc['SM', 'g1'], 100.0)
        self.assertEqual(tables["Offline RPGO Runtime (s)"].loc['SM', 'g1'], 1000.0)

    def test_missing_group_is_empty_cell(self):
        # ROMAN has no g2: its cell is empty (NaN) in every table and its average skips it, without raising
        tables = {}
        def capture(raw_df, title, **kwargs):
            tables[title] = raw_df
            return unittest.mock.MagicMock()  # stands in for the table object whose .to_latex is called
        slam_data_by_run = {'SM': {'g1': self._group(self.SM_ALIGN), 'g2': self._group(self.SM_ALIGN)},
                            'ROMAN': {'g1': self._group(self.ROMAN_ALIGN)}}
        with unittest.mock.patch.object(SLAMEvaluator, 'make_highlighted_table', side_effect=capture), \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_timing_table(['SM', 'ROMAN'], ['g1', 'g2'], {}, slam_data_by_run, Path(tmp_dir) / 'timing.pdf')

        for title, table in tables.items():
            with self.subTest(table=title):
                self.assertTrue(np.isnan(table.loc['ROMAN', 'g2']))
                self.assertFalse(np.isnan(table.loc['SM', 'g2']))
                self.assertEqual(table.loc['ROMAN', 'Average'], table.loc['ROMAN', 'g1'])

    def test_tex_holds_only_uncolored_alignment_table(self):
        slam_data_by_run = {'SM': {'g1': self._group(self.SM_ALIGN)}, 'ROMAN': {'g1': self._group(self.ROMAN_ALIGN)}}
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_timing_table(['SM', 'ROMAN'], ['g1'], {}, slam_data_by_run, Path(tmp_dir) / 'timing.pdf')
            with open(Path(tmp_dir) / 'timing.tex') as f:
                content = f.read()

        self.assertEqual(content.count(r"\begin{table*}"), 1)
        self.assertIn(r"\caption{Alignment Runtime (s).}", content)
        self.assertIn(r"    \textbf{Method} & \textbf{g1} & \textbf{Average} \\", content)
        self.assertIn(r"    SM & \textbf{33.0} & \textbf{33.0} \\", content)
        self.assertIn(r"    ROMAN & \underline{73.0} & \underline{73.0} \\", content)
        for other_value in ("100.0", "1000.0", "1133.0", "1173.0"):  # mapping, RPGO, and totals stay out
            self.assertNotIn(other_value, content)
        self.assertNotIn(r"\textcolor", content)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveDataSizeTable(unittest.TestCase):
    """SLAMEvaluator._save_data_size_table: the data size table uses each group's data_size_mb and the objects table
    its data_num_objects_sent, with an empty cell where objects sent weren't recorded."""

    def _group(self, data_size_mb: float, data_num_objects_sent) -> SLAMData:
        slam_data = object.__new__(SLAMData)
        slam_data.data_size_mb = data_size_mb
        slam_data.data_num_objects_sent = data_num_objects_sent
        return slam_data

    def _capture_tables(self, slam_data_by_run: dict) -> dict:
        tables = {}
        def capture(raw_df, title, **kwargs):
            tables[title] = raw_df
            return unittest.mock.MagicMock()  # stands in for the table object whose .to_latex is called
        with unittest.mock.patch.object(SLAMEvaluator, 'make_highlighted_table', side_effect=capture), \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_data_size_table(['SM', 'ROMAN'], ['g1'], {}, slam_data_by_run, Path(tmp_dir) / 'data_size.pdf')
        return tables

    def test_tables_use_loaded_totals(self):
        tables = self._capture_tables({'SM': {'g1': self._group(3.0, 30)}, 'ROMAN': {'g1': self._group(5.0, 50)}})
        size, objects = tables["Estimated Communication Data Size (MB)"], tables["Total Number of Objects Sent"]
        self.assertEqual((size.loc['SM', 'g1'], objects.loc['SM', 'g1']), (3.0, 30))
        self.assertEqual((size.loc['ROMAN', 'g1'], objects.loc['ROMAN', 'g1']), (5.0, 50))

    def test_unrecorded_objects_is_empty_cell(self):
        tables = self._capture_tables({'SM': {'g1': self._group(3.0, 30)}, 'ROMAN': {'g1': self._group(5.0, None)}})
        self.assertTrue(np.isnan(tables["Total Number of Objects Sent"].loc['ROMAN', 'g1']))
        self.assertEqual(tables["Estimated Communication Data Size (MB)"].loc['ROMAN', 'g1'], 5.0)

    def test_tex_holds_both_uncolored_tables_in_order(self):
        slam_data_by_run = {'SM': {'g1': self._group(3.0, 30)}, 'ROMAN': {'g1': self._group(5.0, 50)}}
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_data_size_table(['SM', 'ROMAN'], ['g1'], {}, slam_data_by_run, Path(tmp_dir) / 'data_size.pdf')
            with open(Path(tmp_dir) / 'data_size.tex') as f:
                content = f.read()

        self.assertEqual(content.count(r"\begin{table*}"), 2)
        size_tex, objects_tex = content.split("\n\n" + r"\begin{table*}")
        self.assertIn(r"\label{tab:data_size}", size_tex)
        self.assertIn(r"    SM & \textbf{3.00} & \textbf{3.00} \\", size_tex)
        self.assertIn(r"    ROMAN & \underline{5.00} & \underline{5.00} \\", size_tex)
        self.assertIn(r"\label{tab:num_objects_sent}", objects_tex)
        self.assertIn(r"    SM & \textbf{30} & \textbf{30} \\", objects_tex)
        self.assertIn(r"    ROMAN & \underline{50} & \underline{50} \\", objects_tex)
        self.assertNotIn(r"\textcolor", content)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestStyleCombinedColumns(unittest.TestCase):
    """SLAMEvaluator._style_combined_columns colors its separator by the given style."""

    def _separator(self, **kwargs) -> TableData.FormattedTextSegment:
        df = pd.DataFrame({'g1': [1.0]}, index=['R'])
        return SLAMEvaluator._style_combined_columns(df, df, "T", fmt=TableData.fmt_fixed(0), **kwargs).df.iat[0, 0][1]

    def test_default_separator_is_georgia_tech_colored(self):
        self.assertEqual(self._separator(), TableData.FormattedTextSegment('/', color='#1A3055'))

    def test_latex_separator_is_uncolored(self):
        self.assertEqual(self._separator(style=TableData.TableStyleName.LATEX, separator=' '),
                         TableData.FormattedTextSegment(' ', color=None))

    def _styles(self, **kwargs) -> list:
        """Returns ``[(A styles, B styles)]`` per row for A = [5, 3] and B = [10, 20]."""
        df_a = pd.DataFrame({'g1': [5.0, 3.0]}, index=['R', 'M'])
        df_b = pd.DataFrame({'g1': [10.0, 20.0]}, index=['R', 'M'])
        table = SLAMEvaluator._style_combined_columns(df_a, df_b, "T", fmt=TableData.fmt_fixed(0), **kwargs)
        return [(table.df.iat[row, 0][0].styles, table.df.iat[row, 0][2].styles) for row in range(2)]

    def test_only_first_value_is_ranked(self):
        self.assertEqual(self._styles(), [([TableData.TextStyle.BOLD], []), ([TableData.TextStyle.UNDERLINE], [])])

    def test_highlight_false_ranks_neither_value(self):
        self.assertEqual(self._styles(highlight=False), [([], []), ([], [])])


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveRobustnessTable(unittest.TestCase):
    """SLAMEvaluator._save_robustness_table: one "failures (rate %)" cell per run and dataset, failing on ATE strictly
    above the threshold or a suppressed multi-robot cell, with each dataset subheaded by its group count."""

    def _result(self, ate: float, num_inter_lc: int) -> SLAMResult:
        merged_metrics = unittest.mock.MagicMock()
        merged_metrics.APE.translation_part.rmse = ate
        return SLAMResult(None, merged_metrics, [], [], [],
                          lc_stats_by_mode={LoopClosureFilterMode.ONLY_INTER_LC: {'num_loop_closures': num_inter_lc}})

    def setUp(self):
        # Datasets interleave (DsA, DsB, DsA, DsB) to check first-appearance column order; DsB's singleton makes it "groups"
        self.groups = [RobotGroup(('a', 'b'), 'DsA', 's', 'g1', None), RobotGroup(('c', 'd'), 'DsB', 's', 'g2', None),
                       RobotGroup(('a', 'c'), 'DsA', 's', 'g3', None), RobotGroup(('e',), 'DsB', 's', 'g4', None)]
        self.results = {
            # g1 above threshold fails, g3 exactly at threshold passes, singleton g4 above threshold fails
            'R': {'g1': self._result(25.0, 3), 'g2': self._result(1.0, 3), 'g3': self._result(20.0, 3), 'g4': self._result(99.0, 0)},
            # g2 suppressed (no inter-robot LC) fails; singleton g4 with no LC is never suppressed, so passes
            'M': {'g1': self._result(1.0, 3), 'g2': self._result(1.0, 0), 'g3': self._result(1.0, 3), 'g4': self._result(1.0, 0)},
            'H': {'g1': self._result(30.0, 3), 'g2': self._result(1.0, 3), 'g3': self._result(40.0, 3), 'g4': self._result(1.0, 0)},
        }
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.save_path = Path(self.tmp_dir.name) / 'robustness_table.pdf'
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf') as to_pdf:
            SLAMEvaluator._save_robustness_table(['R', 'M', 'H'], self.groups, {'g1', 'g2', 'g3'}, {'R': 'ROMAN [6]'},
                                                 self.results, self.save_path, dict.fromkeys(['g1', 'g2', 'g3', 'g4'], 20.0))
        self.pdf_tables = to_pdf.call_args[0][0]
        with open(self.save_path.with_suffix('.tex')) as f:
            self.tex_lines = f.read().split("\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_tex_cells_ranked_lower_is_better(self):
        self.assertIn(r"    ROMAN [6] & \underline{1} \underline{(50.0\%)} & \underline{1} \underline{(50.0\%)} \\", self.tex_lines)
        self.assertIn(r"    M & \textbf{0} \textbf{(0.0\%)} & \underline{1} \underline{(50.0\%)} \\", self.tex_lines)
        self.assertIn(r"    H & 2 (100.0\%) & \textbf{0} \textbf{(0.0\%)} \\", self.tex_lines)

    def test_tex_layout_matches_paper_table(self):
        header_idx = self.tex_lines.index(r"    \textbf{Method} & \textbf{DsA} & \textbf{DsB} \\")
        self.assertEqual(self.tex_lines[header_idx + 1:header_idx + 3],
                         [r"    \cline{2-3}", r"     & \textit{2 pairs} & \textit{2 groups} \\"])
        self.assertEqual(self.tex_lines[0], r"\begin{table}[ht]")
        self.assertIn(r"    \begin{tabular}{||l|c|c||}", self.tex_lines)
        self.assertIn(r"    \caption{Alignment failures. Best bold, second underlined.}", self.tex_lines)
        self.assertFalse(any(r"\textcolor" in line for line in self.tex_lines))

    def test_pdf_is_one_table_with_stacked_subheaders(self):
        self.assertEqual(len(self.pdf_tables), 1)
        self.assertEqual(list(self.pdf_tables[0].df.columns), ["DsA\n2 pairs", "DsB\n2 groups"])
        self.assertEqual([seg.text for seg in self.pdf_tables[0].df.iat[2, 0]], ["2", " ", "(100.0%)"])

    def test_each_group_fails_against_its_own_threshold(self):
        # R: g1 25 <= 30 passes, g2 1 > 0.5 fails, g3 20 > 10 fails, g4 99 <= 100 passes
        # M: only suppressed g2 fails; H: g1 exactly 30 passes, g2 and g3 fail (a shared 20 would give H 2 / 0 instead)
        thresholds = {'g1': 30.0, 'g2': 0.5, 'g3': 10.0, 'g4': 100.0}
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf') as to_pdf:
            SLAMEvaluator._save_robustness_table(['R', 'M', 'H'], self.groups, {'g1', 'g2', 'g3'}, {},
                                                 self.results, self.save_path, thresholds)
        table = to_pdf.call_args[0][0][0]
        failures = [[table.df.iat[row, col][0].text for col in range(2)] for row in range(3)]
        self.assertEqual(failures, [["1", "1"], ["0", "1"], ["1", "1"]])

    def test_missing_group_threshold_raises(self):
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf'), \
             self.assertRaises(KeyError) as ctx:
            SLAMEvaluator._save_robustness_table(['R', 'M', 'H'], self.groups, {'g1', 'g2', 'g3'}, {},
                                                 self.results, self.save_path, dict.fromkeys(['g1', 'g2', 'g3'], 20.0))
        self.assertIn('g4', str(ctx.exception))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMakeHighlightedTable(unittest.TestCase):
    """SLAMEvaluator.make_highlighted_table forwards color_fn_by_col and still ranks every column."""

    def test_color_fn_by_col_forwarded(self):
        style = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
        raw_df = pd.DataFrame({'ColA': {'r1': 1.0, 'r2': 2.0}, 'ColB': {'r1': 3.0, 'r2': 4.0}})
        table = SLAMEvaluator.make_highlighted_table(raw_df, "T", fmt=TableData.fmt_fixed(1), higher_is_better=False,
                    color_fn_by_col={'ColB': TableData.color_fn_NAVY_RED_missing_or_above(3.5)})
        colors = [[table.df.iat[r, c][0].color for c in range(2)] for r in range(2)]
        self.assertEqual(colors, [[style.TextColor, style.TextColor], [style.TextColor, style.TextFailureColor]])
        self.assertIn(TableData.TextStyle.BOLD, table.df.iat[0, 0][0].styles)
        self.assertIn(TableData.TextStyle.BOLD, table.df.iat[0, 1][0].styles)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveAteTablesPerGroupThresholds(unittest.TestCase):
    """SLAMEvaluator._save_ate_tables reds each ATE column by its own group's threshold (with "Average" red only when
    NaN), reds absolute rotation by rot_threshold_deg, and reds the relative-error tables only when NaN; the LaTeX ATE
    table follows the same ATE thresholds."""

    GT = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)
    LATEX = TableData.get_table_style(TableData.TableStyleName.LATEX)
    ATE_TITLES = ["Merged RMS ATE (m) — Pre-Optimize", "Merged RMS ATE (m)"]
    RELATIVE_TITLES = ["Merged RMS RTE (m) - Δ5m", "Merged RMS Relative Rotation Error (deg) - Δ5m"]

    @staticmethod
    def _result(trans: float, rot: float, num_inter_lc: int) -> SLAMResult:
        metrics = unittest.mock.MagicMock()
        metrics.APE.translation_part.rmse = trans
        metrics.RPE.translation_part.rmse = trans
        metrics.APE.rotation_angle_deg.rmse = rot
        metrics.RPE.rotation_angle_deg.rmse = rot
        return SLAMResult(metrics, metrics, [], [], [],
                          lc_stats_by_mode={LoopClosureFilterMode.ONLY_INTER_LC: {'num_loop_closures': num_inter_lc}})

    @staticmethod
    def _colors(table: TableData) -> dict:
        return {(row, col): table.df.at[row, col][0].color for row in table.df.index for col in table.df.columns}

    def _run(self, thresholds: dict):
        # g1 threshold 1, g2 threshold 20, rotation threshold 10; S has no inter-robot LC, so every S cell is suppressed
        results = {'R': {'g1': self._result(5.0, 5.0, 3), 'g2': self._result(20.0, 15.0, 3)},
                   'M': {'g1': self._result(0.5, 5.0, 3), 'g2': self._result(50.0, 15.0, 3)},
                   'S': {'g1': self._result(1.0, 1.0, 0), 'g2': self._result(1.0, 1.0, 0)}}
        pdf_colors = {}
        def snapshot(tables, path, **kwargs):
            for table in tables:
                pdf_colors[table.df.attrs["title"]] = self._colors(table)
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf', side_effect=snapshot) as to_pdf, \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_ate_tables(['R', 'M', 'S'], ['g1', 'g2'], {'g1', 'g2'}, {}, results,
                                           Path(tmp_dir) / 'metrics_table.pdf', thresholds, 10.0)
        return pdf_colors, self._colors(to_pdf.call_args[0][0][1])

    @staticmethod
    def _expected(r_colors: tuple, m_colors: tuple, red: str, navy: str) -> dict:
        # Every table's Average is navy for R and M, and S (all suppressed) is NaN, so red, everywhere
        return {('R', 'g1'): r_colors[0], ('R', 'g2'): r_colors[1], ('R', 'Average'): navy,
                ('M', 'g1'): m_colors[0], ('M', 'g2'): m_colors[1], ('M', 'Average'): navy,
                ('S', 'g1'): red, ('S', 'g2'): red, ('S', 'Average'): red}

    def _expected_ate(self, red: str, navy: str) -> dict:
        # R: g1 5 > 1 red, g2 exactly 20 navy; M: g1 0.5 navy, g2 50 > 20 red; Averages 12.5 / 25.25 never red
        return self._expected((red, navy), (navy, red), red, navy)

    def test_ate_tables_use_per_group_thresholds(self):
        pdf_colors, _ = self._run({'g1': 1.0, 'g2': 20.0})
        for title in self.ATE_TITLES:
            with self.subTest(table=title):
                self.assertEqual(pdf_colors[title], self._expected_ate(self.GT.TextFailureColor, self.GT.TextColor))

    def test_absolute_rotation_uses_rot_threshold(self):
        # Threshold 10: g1 5 navy, g2 15 red, Average exactly 10 navy -- the opposite of g1/g2's ATE thresholds
        red, navy = self.GT.TextFailureColor, self.GT.TextColor
        pdf_colors, _ = self._run({'g1': 1.0, 'g2': 20.0})
        self.assertEqual(pdf_colors["Merged RMS Absolute Rotation Error (deg)"],
                         self._expected((navy, red), (navy, red), red, navy))

    def test_relative_tables_red_only_when_nan(self):
        # R g1 = 5 would be red under g1's ATE threshold, and g2 = 15 under rot_threshold_deg; neither applies here
        red, navy = self.GT.TextFailureColor, self.GT.TextColor
        pdf_colors, _ = self._run({'g1': 1.0, 'g2': 20.0})
        for title in self.RELATIVE_TITLES:
            with self.subTest(table=title):
                self.assertEqual(pdf_colors[title], self._expected((navy, navy), (navy, navy), red, navy))

    def test_latex_ate_table_uses_per_group_thresholds(self):
        _, latex_colors = self._run({'g1': 1.0, 'g2': 20.0})
        self.assertEqual(latex_colors, self._expected_ate(self.LATEX.TextFailureColor, self.LATEX.TextColor))

    def test_extra_threshold_ignored(self):
        pdf_colors, _ = self._run({'g1': 1.0, 'g2': 20.0, 'g9': 0.1})
        self.assertEqual(pdf_colors["Merged RMS ATE (m)"], self._expected_ate(self.GT.TextFailureColor, self.GT.TextColor))

    def test_missing_threshold_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self._run({'g1': 1.0})
        self.assertIn('g2', str(ctx.exception))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveAteSplitTablePerGroupThresholds(unittest.TestCase):
    """SLAMEvaluator._save_ate_split_table reds each robot's ATE column by its own group's threshold, and its RPE
    columns only when NaN."""

    GT = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)

    @staticmethod
    def _metrics(trans: float) -> unittest.mock.MagicMock:
        metrics = unittest.mock.MagicMock()
        metrics.APE.translation_part.rmse = trans
        metrics.RPE.translation_part.rmse = trans
        return metrics

    def _run(self, thresholds: dict) -> list:
        groups = [RobotGroup(('a', 'b'), 'ds', 's', 'g1', None), RobotGroup(('c', 'd'), 'ds', 's', 'g2', None)]
        # d has no metrics, so its cell is NaN
        results = {'R': {'g1': SLAMResult(None, None, [self._metrics(5.0), self._metrics(0.5)], [], []),
                         'g2': SLAMResult(None, None, [self._metrics(10.0), None], [], [])}}
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf') as to_pdf, \
             tempfile.TemporaryDirectory() as tmp_dir:
            SLAMEvaluator._save_ate_split_table(['R'], groups, {}, results, Path(tmp_dir) / 'split.pdf', thresholds)
        return to_pdf.call_args[0][0]

    def test_ate_columns_use_their_group_threshold_and_rpe_red_only_when_nan(self):
        # ATE -- g1 threshold 1: a 5 red, b 0.5 navy; g2 threshold 10: c exactly 10 navy, d NaN red
        # RPE -- a 5 would be red under g1's ATE threshold, but only d (NaN) is red
        red, navy = self.GT.TextFailureColor, self.GT.TextColor
        ate_table, rpe_table = self._run({'g1': 1.0, 'g2': 10.0})
        expected = {"Individual RMS ATE (m)": (ate_table, {'g1\na': red, 'g1\nb': navy, 'g2\nc': navy, 'g2\nd': red}),
                    "Individual RMS RPE (m) - Δ5m": (rpe_table, {'g1\na': navy, 'g1\nb': navy, 'g2\nc': navy, 'g2\nd': red})}
        for title, (table, expected_colors) in expected.items():
            with self.subTest(table=title):
                self.assertEqual(table.df.attrs["title"], title)
                self.assertEqual({col: table.df.at['R', col][0].color for col in table.df.columns}, expected_colors)

    def test_missing_group_threshold_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self._run({'g1': 1.0})
        self.assertIn('g2', str(ctx.exception))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveFailureThresholdTable(unittest.TestCase):
    """SLAMEvaluator._save_failure_threshold_table: one unranked, uncolored-by-threshold "Threshold" row of each
    group's threshold, in cols order, saved as a single-table PDF."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.save_path = Path(self.tmp_dir.name) / 'not_yet_created' / 'failure_thresholds.pdf'

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _run(self, cols: List[str], thresholds: dict):
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf') as to_pdf:
            SLAMEvaluator._save_failure_threshold_table(cols, thresholds, self.save_path)
        return to_pdf

    def test_columns_follow_cols_order_with_rounded_values(self):
        # The dict lists g1 first, cols lists g2 first
        df = self._run(['g2', 'g1'], {'g1': 1.234, 'g2': 56.789}).call_args[0][0][0].df
        self.assertEqual(list(df.columns), ['g2', 'g1'])
        self.assertEqual(list(df.index), ["Threshold"])
        self.assertEqual([df.iat[0, col][0].text for col in range(2)], ["56.79", "1.23"])
        self.assertEqual(df.attrs["title"], "RMS ATE Failure Threshold (m)")

    def test_cells_navy_and_unranked(self):
        navy = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH).TextColor
        df = self._run(['g1', 'g2', 'g3'], {'g1': 1.0, 'g2': 1e6, 'g3': 0.5}).call_args[0][0][0].df
        self.assertEqual([df.iat[0, col][0].color for col in range(3)], [navy] * 3)
        self.assertEqual([df.iat[0, col][0].styles for col in range(3)], [[]] * 3)

    def test_saves_single_table_to_path_creating_parent(self):
        to_pdf = self._run(['g1'], {'g1': 2.0})
        to_pdf.assert_called_once()
        tables, path = to_pdf.call_args[0]
        self.assertEqual((len(tables), path), (1, str(self.save_path)))
        self.assertTrue(self.save_path.parent.is_dir())

    def test_missing_threshold_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self._run(['g2', 'g1'], {'g2': 2.0})
        self.assertIn('g1', str(ctx.exception))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveLCTables(unittest.TestCase):
    """SLAMEvaluator._save_lc_tables: PDF tables titled by precision, plus a .tex of only the all-LC precision and
    successful/total tables, written for every LC filter mode."""

    def setUp(self):
        mode = LoopClosureFilterMode.ONLY_INTER_LC
        all_lc = {'R': (40.0, 2, 5), 'M': (80.0, 4, 5), 'Z': (0.0, 0, 3)}
        inlier_lc = {'R': (11.0, 1, 9), 'M': (22.0, 2, 9), 'Z': (33.0, 3, 9)}  # distinct values that must stay out of the .tex
        def stats(rate: float, successful: int, total: int) -> dict:
            return {'success_rate': rate, 'num_successful_loop_closures': successful, 'num_loop_closures': total}
        results = {run: {'g1': SLAMResult(None, None, [], [], [],
                                          lc_stats_by_mode={mode: stats(*all_lc[run])},
                                          lc_inlier_stats_by_mode={mode: stats(*inlier_lc[run])})}
                   for run in all_lc}
        self.tmp_dir = tempfile.TemporaryDirectory()
        save_path = Path(self.tmp_dir.name) / 'lc_tables.pdf'
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.TableData.to_pdf') as to_pdf:
            SLAMEvaluator._save_lc_tables(['R', 'M', 'Z'], {}, results, mode, save_path)
        self.pdf_tables = to_pdf.call_args[0][0]
        with open(save_path.with_suffix('.tex')) as f:
            self.content = f.read()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_pdf_titles_use_precision(self):
        self.assertEqual([table.df.attrs["title"] for table in self.pdf_tables],
                         ["LC Precision", "LC Successful / Total", "Inlier LC Precision", "Inlier LC Successful / Total"])

    def test_tex_precision_table(self):
        precision_tex, _ = self.content.split("\n\n" + r"\begin{table*}")
        self.assertIn(r"\label{tab:lc_precision_only_inter_lc}", precision_tex)
        self.assertIn(r"    R & \underline{40.0\%} & \underline{40.0\%} \\", precision_tex)
        self.assertIn(r"    M & \textbf{80.0\%} & \textbf{80.0\%} \\", precision_tex)
        self.assertIn(r"    Z & \textcolor[HTML]{CC2222}{0.0\%} & \textcolor[HTML]{CC2222}{0.0\%} \\", precision_tex)

    def test_tex_successful_total_table_has_uncolored_separator(self):
        _, counts_tex = self.content.split("\n\n" + r"\begin{table*}")
        self.assertIn(r"\label{tab:lc_successful_total_only_inter_lc}", counts_tex)
        self.assertIn(r"    R & \underline{2}/5 \\", counts_tex)
        self.assertIn(r"    M & \textbf{4}/5 \\", counts_tex)
        self.assertIn(r"    Z & \textcolor[HTML]{CC2222}{0}/3 \\", counts_tex)

    def test_tex_excludes_inlier_tables(self):
        self.assertEqual(self.content.count(r"\begin{table*}"), 2)
        for inlier_value in ("11.0", "22.0", "33.0", "/9"):
            self.assertNotIn(inlier_value, self.content)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestComputeFailureThreshold(unittest.TestCase):
    """SLAMEvaluator.compute_failure_threshold: extent_fraction of the largest distance between any two GT positions,
    pooled across all robots."""

    @staticmethod
    def _make_gt(positions: List[List[float]]) -> OdometryData:
        n = len(positions)
        return OdometryData('map', 'robot', np.arange(n, dtype=float), np.array(positions, dtype=float),
                            np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), CoordinateFrame.NONE)

    def test_pools_robots(self):
        # Each robot alone spans < 3 m (and they have different counts); the farthest pair (0, 0, 0)-(30, 40, 0) crosses robots
        gt_a = self._make_gt([[0, 0, 0], [1, 1, 0], [2, 1, 0], [1, 2, 0]])
        gt_b = self._make_gt([[30, 40, 0], [29, 39, 0], [29, 40, 0]])
        self.assertAlmostEqual(SLAMEvaluator.compute_failure_threshold([gt_a, gt_b], 0.25), 12.5)

    def test_single_robot(self):
        gt = self._make_gt([[0, 0, 0], [6, 8, 0], [12, 16, 0], [3, 1, 0]])
        self.assertAlmostEqual(SLAMEvaluator.compute_failure_threshold([gt], 0.10), 2.0)

    def test_non_positive_fraction_raises(self):
        gt = self._make_gt([[0, 0, 0], [6, 8, 0]])
        for fraction in (0.0, -0.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                SLAMEvaluator.compute_failure_threshold([gt], fraction)

    def test_empty_gt_list_raises(self):
        with self.assertRaises(ValueError):
            SLAMEvaluator.compute_failure_threshold([], 0.10)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestIsFailure(unittest.TestCase):
    """SLAMEvaluator.is_failure: a NaN (missing/suppressed) ATE, or one strictly above the threshold, is a failure."""

    def test_cases(self):
        cases = [(float('nan'), 10.0, True), (9.9, 10.0, False), (10.0, 10.0, False), (10.1, 10.0, True), (0.0, 0.0, False)]
        for ate, threshold_m, expected in cases:
            with self.subTest(ate=ate, threshold_m=threshold_m):
                self.assertIs(SLAMEvaluator.is_failure(ate, threshold_m), expected)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMissingInterRobotLoopClosures(unittest.TestCase):
    """SLAMEvaluator.missing_inter_robot_loop_closures: only a multi-robot group with zero inter-robot LC qualifies."""

    def test_cases(self):
        cases = [(0, True, True), (1, True, False), (0, False, False), (3, False, False)]
        for num_inter_lc, is_multi_robot, expected in cases:
            with self.subTest(num_inter_lc=num_inter_lc, is_multi_robot=is_multi_robot):
                self.assertIs(SLAMEvaluator.missing_inter_robot_loop_closures(num_inter_lc, is_multi_robot), expected)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestMetricOrNanIfSuppressedMissingData(unittest.TestCase):
    """SLAMEvaluator._metric_or_nan_if_suppressed raises for a missing result, and for missing ONLY_INTER_LC stats only
    on a multi-robot group (a single-robot group never needs them)."""

    @staticmethod
    def _result_without_stats(ate: float) -> SLAMResult:
        merged_metrics = unittest.mock.MagicMock()
        merged_metrics.APE.translation_part.rmse = ate
        return SLAMResult(None, merged_metrics, [], [], [], lc_stats_by_mode={})

    def _metric(self, results: dict, multi_robot_cols: set) -> float:
        return SLAMEvaluator._metric_or_nan_if_suppressed(results, 'R', 'g1', multi_robot_cols,
                                                          attrgetter("merged_metrics.APE.translation_part.rmse"))

    def test_missing_result_raises(self):
        for multi_robot_cols in ({'g1'}, set()):
            with self.subTest(multi_robot_cols=multi_robot_cols), self.assertRaises(ValueError) as ctx:
                self._metric({'R': {'g2': self._result_without_stats(1.0)}}, multi_robot_cols)
            self.assertIn("run='R', col='g1'", str(ctx.exception))

    def test_missing_stats_raises_only_for_multi_robot_group(self):
        results = {'R': {'g1': self._result_without_stats(7.0)}}
        with self.assertRaises(ValueError) as ctx:
            self._metric(results, {'g1'})
        self.assertIn("ONLY_INTER_LC", str(ctx.exception))
        self.assertEqual(self._metric(results, set()), 7.0)


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestSaveLCContextFigureAteTable(unittest.TestCase):
    """SLAMEvaluator._save_lc_context_figure's ATE table: NaN for a multi-robot group with zero inter-robot LC (never for a
    single-robot group), red above the group's threshold, and an error for missing results, stats, or LC counts."""

    GT = TableData.get_table_style(TableData.TableStyleName.GEORGIA_TECH)

    @staticmethod
    def _result(ate: float, lc_stats_by_mode: dict) -> SLAMResult:
        merged_metrics = unittest.mock.MagicMock()
        merged_metrics.APE.translation_part.rmse = ate
        return SLAMResult(None, merged_metrics, [], [], [], lc_stats_by_mode=lc_stats_by_mode)

    def _ate_cells(self, group: tuple, col: str, u_results: Optional[dict] = None) -> dict:
        # R 5 below threshold 10, M 15 above it, S has zero inter-robot LC; a malformed run U is added only when given
        run_names = ['R', 'M', 'S']
        lc_stats = {num: {LoopClosureFilterMode.ONLY_INTER_LC: {'num_loop_closures': num}} for num in (0, 3)}
        results = {'R': {col: self._result(5.0, lc_stats[3])}, 'M': {col: self._result(15.0, lc_stats[3])},
                   'S': {col: self._result(1.0, lc_stats[0])}}
        if u_results is not None:
            run_names.append('U')
            results['U'] = u_results
        stats = [{'success_rate': 50.0, 'num_successful_loop_closures': 1, 'num_loop_closures': 2}] * (2 * len(run_names))
        tables = []
        real_make_highlighted_table = SLAMEvaluator.make_highlighted_table
        def capture(*args, **kwargs):
            tables.append(real_make_highlighted_table(*args, **kwargs))
            return tables[-1]
        with unittest.mock.patch.object(SLAMEvaluator, 'make_highlighted_table', side_effect=capture), \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.LoopClosureData.visualize_error_scatter'), \
             tempfile.TemporaryDirectory() as tmp_dir:
            num_entries = 2 * len(run_names)
            SLAMEvaluator._save_lc_context_figure(group, col, [unittest.mock.MagicMock()] * num_entries, ['x'] * num_entries,
                                                  [i for i in range(len(run_names)) for _ in range(2)],
                                                  stats, results, run_names, Path(tmp_dir), 10.0)
        df = next(t.df for t in tables if list(t.df.columns) == ["RMS ATE (m)"])
        return {run: (df.at[run, "RMS ATE (m)"][0].text, df.at[run, "RMS ATE (m)"][0].color) for run in run_names}

    def test_multi_robot_group(self):
        red, navy = self.GT.TextFailureColor, self.GT.TextColor
        self.assertEqual(self._ate_cells(('a', 'b'), 'g1'), {'R': ("5.00", navy), 'M': ("15.00", red), 'S': ("---", red)})

    def test_single_robot_group_never_suppressed(self):
        red, navy = self.GT.TextFailureColor, self.GT.TextColor
        self.assertEqual(self._ate_cells(('a',), 'g0'), {'R': ("5.00", navy), 'M': ("15.00", red), 'S': ("1.00", navy)})

    def test_missing_result_or_stats_raise(self):
        for group, col in ((('a', 'b'), 'g1'), (('a',), 'g0')):
            for name, u_results in (("missing result", {}), ("missing stats", {col: self._result(2.0, {})})):
                with self.subTest(group=group, case=name), self.assertRaises(ValueError) as ctx:
                    self._ate_cells(group, col, u_results)
                self.assertIn("run='U'", str(ctx.exception))

    def test_stats_without_count_raise(self):
        for group, col in ((('a', 'b'), 'g1'), (('a',), 'g0')):
            u_results = {col: self._result(2.0, {LoopClosureFilterMode.ONLY_INTER_LC: {'success_rate': 50.0}})}
            with self.subTest(group=group), self.assertRaises(KeyError) as ctx:
                self._ate_cells(group, col, u_results)
            self.assertIn('num_loop_closures', str(ctx.exception))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRunEvaluationFailureThresholds(unittest.TestCase):
    """SLAMEvaluator.run_evaluation computes each group's threshold from the GT it already loads (no extra loads),
    and passes it to every table and to that group's LC-context figures."""

    def setUp(self):
        viz_config = RobotGroupViz(name_map=None, robot_name_to_color={}, image_path=None,
                                   image_x_edge=None, image_extent_offsets=None)
        self.groups = [RobotGroup(('a', 'b'), 'ds', 's1', 'g1', viz_config), RobotGroup(('c', 'd'), 'ds', 's2', 'g2', viz_config)]
        self.methods = [SLAMMethod("M1", "Method 1", "M1", "#000000")]
        self.gt_by_seq = {'s1': [unittest.mock.MagicMock(), unittest.mock.MagicMock()],
                          's2': [unittest.mock.MagicMock(), unittest.mock.MagicMock()]}
        self.thresholds = {'g1': 5.0, 'g2': 50.0}

    def _load_gt(self, dataset_seq: str, robot_names: List[str]) -> list:
        return self.gt_by_seq[dataset_seq]

    def _threshold_for_gt(self, gt_data_lst: list, extent_fraction: float) -> float:
        return 5.0 if gt_data_lst is self.gt_by_seq['s1'] else 50.0

    def test_thresholds_computed_once_per_group_and_threaded_through(self):
        load_gt_data_fn = unittest.mock.MagicMock(side_effect=self._load_gt)
        slam_data = unittest.mock.MagicMock()
        slam_data.get_loop_closures.return_value = (unittest.mock.MagicMock(), unittest.mock.MagicMock())
        mock = unittest.mock.DEFAULT
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.Pool') as pool, \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.SLAMData') as slam_data_cls, \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.LoopClosureData') as lc_cls, \
             unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.plt'), \
             unittest.mock.patch.multiple(SLAMEvaluator, compute_failure_threshold=mock, save_merged_ate_figures=mock,
                    _save_all_methods_traj_figure=mock, _save_timing_table=mock, _save_data_size_table=mock,
                    _save_mg_match_table=mock, _save_lc_tables=mock, _save_ate_tables=mock, _save_robustness_table=mock,
                    _save_failure_threshold_table=mock, _save_ate_split_table=mock, _save_lc_context_figure=mock,
                    _save_lc_side_by_side_figure=mock, _save_lc_sep_figure=mock, _save_traj_lc_comb_figure=mock) as patched, \
             tempfile.TemporaryDirectory() as tmp_dir:
            pool.return_value.__enter__.return_value.starmap.side_effect = [[slam_data, slam_data],
                                                                            [unittest.mock.MagicMock(), unittest.mock.MagicMock()]]
            slam_data_cls.get_timing_totals.return_value = {'align': 1.0}
            lc_cls.visualize_error_scatter.return_value = (None, [{}, {}])
            patched['compute_failure_threshold'].side_effect = self._threshold_for_gt

            SLAMEvaluator.run_evaluation(Path('/unused'), Path('out'), self.methods, self.groups, {}, Path(tmp_dir),
                                         load_gt_data_fn, failure_extent_fraction=0.25)
            base_dir = Path(tmp_dir) / 'out'

        # R1: one threshold per group, from that group's GT and the given fraction
        self.assertEqual(patched['compute_failure_threshold'].call_args_list,
                         [unittest.mock.call(self.gt_by_seq['s1'], 0.25), unittest.mock.call(self.gt_by_seq['s2'], 0.25)])
        # R2/R3: every threshold-using table gets the per-group dict
        self.assertEqual(patched['_save_ate_tables'].call_args[0][6], self.thresholds)
        self.assertEqual(patched['_save_robustness_table'].call_args[0][6], self.thresholds)
        self.assertEqual(patched['_save_ate_split_table'].call_args[0][5], self.thresholds)
        self.assertEqual(patched['_save_failure_threshold_table'].call_args[0],
                         (['g1', 'g2'], self.thresholds, base_dir / 'failure_thresholds.pdf'))
        # R4: each LC-context figure (3 LC modes x 2 groups) gets its own group's threshold
        context_calls = sorted((c[0][1], c[0][-1]) for c in patched['_save_lc_context_figure'].call_args_list)
        self.assertEqual(context_calls, [('g1', 5.0)] * 3 + [('g2', 50.0)] * 3)
        # R5: GT loaded only where it already was -- once for the all-methods figure plus once per LC mode
        loads_per_seq = [c[0][0] for c in load_gt_data_fn.call_args_list]
        self.assertEqual((loads_per_seq.count('s1'), loads_per_seq.count('s2'), len(loads_per_seq)), (4, 4, 8))

    def test_old_and_positional_threshold_arguments_rejected(self):
        args = (Path('/unused'), Path('out'), self.methods, self.groups, {}, Path('/unused'), unittest.mock.MagicMock())
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.Pool') as pool:
            for name, kwargs, extra_args in (("ate_threshold_m", {'ate_threshold_m': 20.0}, ()), ("positional", {}, (0.10,))):
                with self.subTest(name), self.assertRaises(TypeError):
                    SLAMEvaluator.run_evaluation(*args, *extra_args, **kwargs)
        pool.assert_not_called()


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestRunEvaluationDuplicateMethodNames(unittest.TestCase):
    """SLAMEvaluator.run_evaluation rejects methods sharing a name (they'd overwrite each other's results), naming
    only the duplicated one, before loading any data."""

    def test_duplicate_method_names_raise_before_loading(self):
        viz_config = RobotGroupViz(name_map=None, robot_name_to_color={}, image_path=None,
                                   image_x_edge=None, image_extent_offsets=None)
        robot_groups = [RobotGroup(('a', 'b'), 'ds', 'seq', 'g1', viz_config)]
        methods = [SLAMMethod("MG_SM", "MeronomyGraph (HMO)", "MG_SM", "#0000FF"),
                   SLAMMethod("ROMAN_O", "ROMAN", "ROMAN_O", "#FF0000"),
                   SLAMMethod("MG_SM", "MeronomyGraph (HMO, Max Size 35)", "MG_SM", "#FF8C00",
                              {"submap_align_params.submap_max_size": 35})]
        with unittest.mock.patch('robotdataprocess.eval.SLAMEvaluator.Pool') as pool:
            with self.assertRaises(ValueError) as ctx:
                SLAMEvaluator.run_evaluation(Path('/unused'), Path('out'), methods, robot_groups, {}, Path('/unused'),
                                             unittest.mock.MagicMock(), failure_extent_fraction=0.10)
        self.assertIn("['MG_SM']", str(ctx.exception))
        pool.assert_not_called()


if __name__ == '__main__':
    unittest.main()

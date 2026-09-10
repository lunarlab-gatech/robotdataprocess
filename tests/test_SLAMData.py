import itertools
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from robotdataprocess.data_types.SLAMData import SLAMData
import sys
import tempfile
from typing import Optional
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
        dataset_name: Which dataset this run is for.
        dataset_version: The dataset version/sequence; also used here for the results
            directory layout.
        offline_rpgo_params: Holds ``sparsified``, read by ``SLAMData.load_LC_data``.
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


def _expected_inlier_lc(rpgo_dir: Path, sorted_names, time_subdir: str) -> LoopClosureData:
    """Reproduces SLAMData.load_LC_data's inlier-assembly loop against a chosen time
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
    """Regression test for SLAMData.load_LC_data's inlier LC loading with sparsified=True
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
        system_params = _FakeSystemParams(dataset_name='kimera_multi', dataset_version=self.DATASET_NAME,
                                          method=self.METHOD, sparsified=True)
        _, lc_inlier = SLAMData.load_LC_data(
            self.ROMAN_ROOT, system_params, sorted(self.ROBOT_NAMES), {})

        expected = _expected_inlier_lc(self.RPGO_DIR, sorted(self.ROBOT_NAMES), 'sparse')
        self.assertEqual(lc_inlier.num_loop_closures, expected.num_loop_closures)
        self.assertEqual(_lc_keys(lc_inlier), _lc_keys(expected))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadLCDataROMANNonSparsified(unittest.TestCase):
    """Regression test for SLAMData.load_LC_data's inlier LC loading with sparsified=False
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
        system_params = _FakeSystemParams(dataset_name='hercules', dataset_version=self.DATASET_NAME,
                                          method=self.METHOD, sparsified=False)
        _, lc_inlier = SLAMData.load_LC_data(
            self.ROMAN_ROOT, system_params, sorted(self.ROBOT_NAMES), {})

        expected = _expected_inlier_lc(self.RPGO_DIR, sorted(self.ROBOT_NAMES), 'dense')
        self.assertEqual(lc_inlier.num_loop_closures, expected.num_loop_closures)
        self.assertEqual(_lc_keys(lc_inlier), _lc_keys(expected))


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadLCData(unittest.TestCase):
    """Tests SLAMData.load_LC_data against a small controlled fixture (independent hardcoded
    expectations, not re-derived via load_LC_data's own assembly loop): odom_and_lc.g2o mixes
    an unmarked odometry edge (must be skipped) with LC edges and a deliberate exact duplicate
    (must be pruned); dense/ and sparse/ time files carry different values for the same
    keyframes (to prove the sparsified branch reads the right one); each per-pair inlier file
    carries values distinct from the others and from odom_and_lc.g2o (to catch a wrong-file-read
    bug); and names_override is exercised both unset and explicitly set.
    """

    MG_ROOT = Path(__file__).parent / 'files' / 'test_SLAMData' / 'load_LC_data'
    DATASET_SEQ = 'test_seq'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['robotA', 'robotB']

    # (name_a, name_b, ts_a, ts_b, [px,py,pz], [qx,qy,qz,qw]) as written into dense/odom_and_lc.g2o,
    # after prune_duplicates -- the 5th (duplicate) edge is expected to collapse into the 4th.
    MERGED_LC_ROWS = [
        ('robotA', 'robotA', 0.1, 0.4, [1.111111, -2.222222, 0.333333],
         [0.0000000, 0.5372996, 0.0000000, 0.8433914]),
        ('robotB', 'robotB', 0.5, 0.8, [-3.333333, 4.444444, -0.555555],
         [-0.6427876, 0.0000000, 0.0000000, 0.7660444]),
        ('robotA', 'robotB', 0.3, 0.7, [5.555555, -6.666666, 7.777777],
         [0.0000000, 0.0000000, 0.9396926, 0.3420201]),
    ]

    # (name_a, name_b, [px,py,pz], [qx,qy,qz,qw]) per inlier file -- timestamps depend on
    # dense vs. sparse and are supplied separately per test.
    INLIER_INTRA_A = ('robotA', 'robotA', [0.1, 0.2, -0.3], [0.0000000, 0.0871557, 0.0000000, 0.9961947])
    INLIER_INTRA_B = ('robotB', 'robotB', [1.5, -1.5, 2.0], [0.9848078, 0.0000000, 0.0000000, -0.1736482])
    INLIER_INTER_AB = ('robotA', 'robotB', [-0.7, 0.7, 0.7], [0.0000000, 0.0000000, -0.4226183, 0.9063078])

    @classmethod
    def _system_params(cls, sparsified: bool) -> _FakeSystemParams:
        return _FakeSystemParams(dataset_name='fake_dataset', dataset_version=cls.DATASET_SEQ,
                                 method=cls.METHOD, sparsified=sparsified)

    @staticmethod
    def _row_by_names(lc: LoopClosureData, name_a: str, name_b: str) -> tuple:
        idx = lc.names.index((name_a, name_b))
        return (float(lc.timestamps_a[idx]), float(lc.timestamps_b[idx]),
                np.array(lc.translations[idx], dtype=float), np.array(lc.orientations[idx], dtype=float))

    def test_merged_lc_skips_odometry_and_prunes_duplicate(self):
        merged_lc, _ = SLAMData.load_LC_data(self.MG_ROOT, self._system_params(False), self.ROBOT_NAMES, {})

        self.assertEqual(merged_lc.num_loop_closures, len(self.MERGED_LC_ROWS))
        for name_a, name_b, ts_a, ts_b, translation, orientation in self.MERGED_LC_ROWS:
            got_ts_a, got_ts_b, got_translation, got_orientation = self._row_by_names(merged_lc, name_a, name_b)
            self.assertAlmostEqual(got_ts_a, ts_a, places=9)
            self.assertAlmostEqual(got_ts_b, ts_b, places=9)
            np.testing.assert_allclose(got_translation, translation, atol=1e-6)
            np.testing.assert_allclose(got_orientation, orientation, atol=1e-6)

        # The bogus 999/999/999 odometry-edge translation must not have leaked in anywhere.
        self.assertTrue(np.all(np.abs(np.array(merged_lc.translations, dtype=float)) < 100))

    def test_inlier_lc_uses_dense_timestamps_when_not_sparsified(self):
        _, merged_inlier = SLAMData.load_LC_data(self.MG_ROOT, self._system_params(False), self.ROBOT_NAMES, {})

        expected = [
            (*self.INLIER_INTRA_A, 0.1, 0.3),
            (*self.INLIER_INTRA_B, 0.5, 0.7),
            (*self.INLIER_INTER_AB, 0.2, 0.6),
        ]
        for name_a, name_b, translation, orientation, ts_a, ts_b in expected:
            got_ts_a, got_ts_b, got_translation, got_orientation = self._row_by_names(merged_inlier, name_a, name_b)
            self.assertAlmostEqual(got_ts_a, ts_a, places=9)
            self.assertAlmostEqual(got_ts_b, ts_b, places=9)
            np.testing.assert_allclose(got_translation, translation, atol=1e-6)
            np.testing.assert_allclose(got_orientation, orientation, atol=1e-6)

    def test_inlier_lc_uses_sparse_timestamps_when_sparsified(self):
        _, merged_inlier = SLAMData.load_LC_data(self.MG_ROOT, self._system_params(True), self.ROBOT_NAMES, {})

        expected = [
            (*self.INLIER_INTRA_A, 0.111, 0.333),
            (*self.INLIER_INTRA_B, 0.555, 0.777),
            (*self.INLIER_INTER_AB, 0.222, 0.666),
        ]
        for name_a, name_b, translation, orientation, ts_a, ts_b in expected:
            got_ts_a, got_ts_b, got_translation, got_orientation = self._row_by_names(merged_inlier, name_a, name_b)
            self.assertAlmostEqual(got_ts_a, ts_a, places=9)
            self.assertAlmostEqual(got_ts_b, ts_b, places=9)
            np.testing.assert_allclose(got_translation, translation, atol=1e-6)
            np.testing.assert_allclose(got_orientation, orientation, atol=1e-6)

    def test_names_override_applied(self):
        override = {'a': 'Alpha', 'b': 'Bravo'}
        merged_lc, merged_inlier = SLAMData.load_LC_data(
            self.MG_ROOT, self._system_params(False), self.ROBOT_NAMES, {}, names_override=override)

        self.assertIn(('Alpha', 'Alpha'), merged_lc.names)
        self.assertIn(('Bravo', 'Bravo'), merged_lc.names)
        self.assertIn(('Alpha', 'Bravo'), merged_lc.names)
        self.assertIn(('Alpha', 'Alpha'), merged_inlier.names)
        self.assertIn(('Bravo', 'Bravo'), merged_inlier.names)
        self.assertIn(('Alpha', 'Bravo'), merged_inlier.names)
        # Un-overridden default (robot_names) must not appear once override is supplied.
        self.assertNotIn(('robotA', 'robotA'), merged_lc.names)

    def test_unsorted_robot_names_raises_value_error(self):
        with self.assertRaises(ValueError):
            SLAMData.load_LC_data(self.MG_ROOT, self._system_params(False), ['robotB', 'robotA'], {})


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadTimingData(unittest.TestCase):
    """Tests SLAMData.load_timing_data against a fixture built fresh per test into a
    tempfile.TemporaryDirectory (rather than checked-in fixtures, since ~7 near-duplicate
    directories -- one per missing/empty-file case -- would be pure repetition). Each align/
    mapping runtime file uses the real "label: value" format; the rpgo runtime file uses its
    real plain-numeric-last-line format (no label), preceded by junk/blank lines, so the
    file-specific parsing logic is exercised as it's actually used in production, not a
    simplified stand-in.
    """

    DATASET_SEQ = 'test_seq'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['robotA', 'robotB']

    # Distinct per-entry values so a value/key mixup between pairs or robots would fail.
    ALIGN_AA = 10.5
    ALIGN_AB = 20.25
    ALIGN_BB = 30.125
    MAPPING_A = 5.5
    MAPPING_B = 6.75
    RPGO = 100.25

    @classmethod
    def _system_params(cls) -> _FakeSystemParams:
        return _FakeSystemParams(dataset_name='fake_dataset', dataset_version=cls.DATASET_SEQ,
                                 method=cls.METHOD, sparsified=False)

    def _write_fixture(self, mg_root: Path, skip: Optional[str] = None, blank: Optional[str] = None) -> None:
        """Writes the full valid runtime-file tree under mg_root, except that the file keyed
        by `skip` is omitted entirely, or the file keyed by `blank` is written whitespace-only.
        """
        method_dir = mg_root / 'results' / self.DATASET_SEQ / self.METHOD
        files = {
            'align_AA': (method_dir / 'align' / 'robotA_robotA' / 'robotA_robotA.runtime.txt',
                        f'align_runtime_s: {self.ALIGN_AA:.6f}\n'),
            'align_AB': (method_dir / 'align' / 'robotA_robotB' / 'robotA_robotB.runtime.txt',
                        f'align_runtime_s: {self.ALIGN_AB:.6f}\n'),
            'align_BB': (method_dir / 'align' / 'robotB_robotB' / 'robotB_robotB.runtime.txt',
                        f'align_runtime_s: {self.ALIGN_BB:.6f}\n'),
            'mapping_A': (method_dir / 'mapping' / 'robotA' / 'robotA.runtime.txt',
                        f'mapping_runtime_s: {self.MAPPING_A:.6f}\n'),
            'mapping_B': (method_dir / 'mapping' / 'robotB' / 'robotB.runtime.txt',
                        f'mapping_runtime_s: {self.MAPPING_B:.6f}\n'),
            'rpgo': (method_dir / 'rpgo' / 'robotA_robotB' / 'runtime.txt',
                    f'iteration 1 converged\n\n{self.RPGO:.6f}\n'),
        }
        for key, (path, content) in files.items():
            if key == skip:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('   \n\n  \n' if key == blank else content)

    def test_happy_path_two_robot_group(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root)

            timing = SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})

        dataset_key = ('fake_dataset', self.DATASET_SEQ)
        self.assertEqual(timing['align'], {
            (*dataset_key, 'robotA', 'robotA'): self.ALIGN_AA,
            (*dataset_key, 'robotA', 'robotB'): self.ALIGN_AB,
            (*dataset_key, 'robotB', 'robotB'): self.ALIGN_BB,
        })
        self.assertEqual(timing['mapping'], {
            (*dataset_key, 'robotA'): self.MAPPING_A,
            (*dataset_key, 'robotB'): self.MAPPING_B,
        })
        self.assertEqual(timing['offline_rpgo'], {(*dataset_key, 'robotA', 'robotB'): self.RPGO})

    def test_missing_align_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, skip='align_BB')

            with self.assertRaises(FileNotFoundError) as ctx:
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})
        self.assertIn('robotB_robotB.runtime.txt', str(ctx.exception))

    def test_missing_mapping_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, skip='mapping_A')

            with self.assertRaises(FileNotFoundError) as ctx:
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})
        self.assertIn(str(Path('mapping') / 'robotA' / 'robotA.runtime.txt'), str(ctx.exception))

    def test_missing_rpgo_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, skip='rpgo')

            with self.assertRaises(FileNotFoundError) as ctx:
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})
        self.assertIn(str(Path('rpgo') / 'robotA_robotB' / 'runtime.txt'), str(ctx.exception))

    def test_empty_align_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, blank='align_AB')

            with self.assertRaises(ValueError):
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})

    def test_empty_mapping_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, blank='mapping_B')

            with self.assertRaises(ValueError):
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})

    def test_empty_rpgo_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root, blank='rpgo')

            with self.assertRaises(ValueError):
                SLAMData.load_timing_data(mg_root, self._system_params(), self.ROBOT_NAMES, {})

    def test_unsorted_robot_names_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mg_root = Path(tmp_dir)
            self._write_fixture(mg_root)

            with self.assertRaises(ValueError):
                SLAMData.load_timing_data(mg_root, self._system_params(), ['robotB', 'robotA'], {})


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadEstData(unittest.TestCase):
    """Tests SLAMData.load_est_data against a fixture with distinct, non-trivial per-row
    positions/orientations and irregular (non-uniform, sub-second, whole-second-crossing)
    timestamps for each of two robots -- chosen so that a column swap (e.g. x/y, qw/qx, or
    the CSV's qw,qx,qy,qz vs. the stored (x,y,z,w) quaternion order), a row-order bug, or a
    timestamp-unit/precision bug would all fail at least one assertion below.
    """

    MG_ROOT = Path(__file__).parent / 'files' / 'test_SLAMData' / 'load_est_data'
    DATASET_SEQ = 'test_seq'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['robotA', 'robotB']

    # (seconds, [x, y, z], [qw, qx, qy, qz]) as written in robotA.csv/robotB.csv -- kept here as
    # plain literals (not re-derived from the CSV) so the test has an independent expectation.
    ROBOT_A_ROWS = [
        (0.0, [0.100000, -2.300000, 0.050000], [1.0000000, 0.0000000, 0.0000000, 0.0000000]),
        (1.5, [4.250000, 1.750000, -0.800000], [0.9238795, 0.3826834, 0.0000000, 0.0000000]),
        (3.200000001, [-1.600000, 9.900000, 2.400000], [0.7071068, 0.0000000, 0.7071068, 0.0000000]),
        (7.899999999, [12.340000, -6.100000, -3.300000], [0.9483237, 0.0000000, 0.0000000, -0.3173047]),
    ]
    ROBOT_B_ROWS = [
        (0.25, [-8.500000, 3.100000, 0.900000], [1.0000000, 0.0000000, 0.0000000, 0.0000000]),
        (1.750000003, [2.200000, -4.400000, 5.500000], [0.6620732, 0.3427368, -0.2610764, 0.6132126]),
        (4.000000005, [-3.330000, 7.770000, -1.110000], [0.5826464, -0.7520439, -0.1352615, 0.2768710]),
        (9.999999998, [0.010000, 0.020000, 15.030000], [0.0156483, 0.9315103, 0.3543141, 0.0806550]),
    ]

    @classmethod
    def _system_params(cls) -> _FakeSystemParams:
        return _FakeSystemParams(dataset_name='fake_dataset', dataset_version=cls.DATASET_SEQ,
                                 method=cls.METHOD, sparsified=False)

    def _assert_matches_rows(self, trajectory, rows, expected_child_frame_id: str) -> None:
        self.assertEqual(trajectory.frame_id, 'map')
        self.assertEqual(trajectory.child_frame_id, expected_child_frame_id)
        self.assertEqual(len(trajectory.timestamps), len(rows))

        timestamps_s = [float(t) for t in trajectory.timestamps]
        expected_timestamps_s = [row[0] for row in rows]
        np.testing.assert_allclose(timestamps_s, expected_timestamps_s, atol=1e-9)

        positions = np.array(trajectory.positions, dtype=float)
        expected_positions = np.array([row[1] for row in rows])
        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)

        # trajectory.orientations are stored (x, y, z, w); CSV/expectation rows are (w, x, y, z).
        orientations = np.array(trajectory.orientations, dtype=float)
        expected_quats_wxyz = np.array([row[2] for row in rows])
        expected_quats_xyzw = expected_quats_wxyz[:, [1, 2, 3, 0]]
        np.testing.assert_allclose(orientations, expected_quats_xyzw, atol=1e-6)

    def test_loads_trajectories_in_robot_names_order(self):
        trajectories = SLAMData.load_est_data(self.MG_ROOT, self._system_params(), self.ROBOT_NAMES, {})

        self.assertEqual(len(trajectories), 2)
        self._assert_matches_rows(trajectories[0], self.ROBOT_A_ROWS, 'robot0')
        self._assert_matches_rows(trajectories[1], self.ROBOT_B_ROWS, 'robot1')

    def test_single_robot_group(self):
        trajectories = SLAMData.load_est_data(self.MG_ROOT, self._system_params(), ['robotA'], {})

        self.assertEqual(len(trajectories), 1)
        self._assert_matches_rows(trajectories[0], self.ROBOT_A_ROWS, 'robot0')

    def test_unsorted_robot_names_raises_value_error(self):
        with self.assertRaises(ValueError):
            SLAMData.load_est_data(self.MG_ROOT, self._system_params(), ['robotB', 'robotA'], {})

    def test_missing_csv_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            SLAMData.load_est_data(self.MG_ROOT, self._system_params(), ['robotC'], {})


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestLoadKimeraRPGOFirstStageEstData(unittest.TestCase):
    """Tests SLAMData.load_kimera_rpgo_first_stage_est_data against a fixture with distinct,
    non-trivial per-vertex positions/orientations, irregular (sub-second, fractional-ns,
    whole-second-crossing) timestamps, and g2o vertices written out of keyframe order -- chosen
    so that a names_override char/robot mixup, a missing keyframe-sort, a quaternion-order bug
    (unlike from_csv, from_g2o does *not* reorder qx,qy,qz,qw -- it already matches the stored
    (x,y,z,w) order), or a (robot_id, keyframe_id) timestamp-lookup bug would all fail below.
    """

    MG_ROOT = Path(__file__).parent / 'files' / 'test_SLAMData' / 'load_kimera_rpgo_first_stage_est_data'
    DATASET_SEQ = 'test_seq'
    METHOD = 'ROMAN'
    ROBOT_NAMES = ['robotA', 'robotB']

    # (seconds, [x, y, z], [qx, qy, qz, qw]) in keyframe order, as encoded into result.g2o/
    # odom_all.time.txt -- kept here as independent literals, not re-derived from the fixture.
    ROBOT_A_ROWS = [
        (0.0, [0.300000, -1.200000, 0.400000], [0.1927273, -0.0121613, -0.5036369, 0.8420559]),
        (1.2, [2.100000, 3.300000, -0.900000], [-0.3780207, 0.2461412, 0.1472241, 0.8802499]),
        (2.900000003, [-4.400000, 0.700000, 5.100000], [0.7071068, 0.0000000, 0.0000000, 0.7071068]),
        (8.100000007, [6.600000, -2.200000, -1.300000], [0.2233291, 0.9652812, -0.1204230, 0.0620857]),
    ]
    ROBOT_B_ROWS = [
        (0.5, [-3.100000, 2.400000, 1.900000], [-0.1442003, 0.7517881, 0.1366756, 0.6287612]),
        (1.900000002, [1.100000, -5.500000, 3.300000], [0.0270976, -0.5141426, -0.7717387, 0.3732862]),
        (4.300000009, [7.700000, 0.200000, -6.600000], [0.0000000, -0.7071068, 0.0000000, 0.7071068]),
        (10.500000004, [-0.500000, 8.800000, 2.200000], [0.6950619, 0.6843380, 0.1195751, 0.1851278]),
    ]

    @classmethod
    def _system_params(cls) -> _FakeSystemParams:
        return _FakeSystemParams(dataset_name='fake_dataset', dataset_version=cls.DATASET_SEQ,
                                 method=cls.METHOD, sparsified=False)

    def _assert_matches_rows(self, trajectory, rows, expected_child_frame_id: str) -> None:
        self.assertEqual(trajectory.frame_id, 'map')
        self.assertEqual(trajectory.child_frame_id, expected_child_frame_id)
        self.assertEqual(len(trajectory.timestamps), len(rows))

        timestamps_s = [float(t) for t in trajectory.timestamps]
        expected_timestamps_s = [row[0] for row in rows]
        np.testing.assert_allclose(timestamps_s, expected_timestamps_s, atol=1e-9)

        positions = np.array(trajectory.positions, dtype=float)
        expected_positions = np.array([row[1] for row in rows])
        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)

        # from_g2o's (qx, qy, qz, qw) already matches the stored (x, y, z, w) order -- no reorder.
        orientations = np.array(trajectory.orientations, dtype=float)
        expected_quats_xyzw = np.array([row[2] for row in rows])
        np.testing.assert_allclose(orientations, expected_quats_xyzw, atol=1e-6)

    def test_loads_trajectories_in_robot_names_order(self):
        trajectories = SLAMData.load_kimera_rpgo_first_stage_est_data(
            self.MG_ROOT, self._system_params(), self.ROBOT_NAMES, {})

        self.assertEqual(len(trajectories), 2)
        self._assert_matches_rows(trajectories[0], self.ROBOT_A_ROWS, 'robot0')
        self._assert_matches_rows(trajectories[1], self.ROBOT_B_ROWS, 'robot1')

    def test_single_robot_group(self):
        trajectories = SLAMData.load_kimera_rpgo_first_stage_est_data(
            self.MG_ROOT, self._system_params(), ['robotA'], {})

        self.assertEqual(len(trajectories), 1)
        self._assert_matches_rows(trajectories[0], self.ROBOT_A_ROWS, 'robot0')

    def test_unsorted_robot_names_raises_value_error(self):
        with self.assertRaises(ValueError):
            SLAMData.load_kimera_rpgo_first_stage_est_data(
                self.MG_ROOT, self._system_params(), ['robotB', 'robotA'], {})

    def test_missing_vertices_raises_value_error(self):
        # robotA_robotD's g2o only has char 'a' vertices -- robotD (mapped to char 'b') has none.
        with self.assertRaises(ValueError):
            SLAMData.load_kimera_rpgo_first_stage_est_data(
                self.MG_ROOT, self._system_params(), ['robotA', 'robotD'], {})


class TestAssertSortedRobotNames(unittest.TestCase):
    """Covers SLAMData._assert_sorted_robot_names's raise and pass-through branches."""

    def test_sorted_names_does_not_raise(self):
        SLAMData._assert_sorted_robot_names(['a', 'b', 'c'])

    def test_empty_list_does_not_raise(self):
        SLAMData._assert_sorted_robot_names([])

    def test_single_name_does_not_raise(self):
        SLAMData._assert_sorted_robot_names(['a'])

    def test_unsorted_names_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            SLAMData._assert_sorted_robot_names(['b', 'a'])
        self.assertIn("robot_names must be sorted", str(ctx.exception))
        self.assertIn("['b', 'a']", str(ctx.exception))

    def test_accepts_tuple_of_unsorted_names_raises(self):
        # list(robot_names) coerces non-list iterables, so a tuple must be checked too.
        with self.assertRaises(ValueError):
            SLAMData._assert_sorted_robot_names(('c', 'a', 'b'))


class TestEnsureMeronomyGraphImportable(unittest.TestCase):
    """Covers SLAMData.ensure_MeronomyGraph_importable's already-present and not-yet-present
    branches of the sys.path membership check.
    """

    def setUp(self):
        self._original_sys_path = list(sys.path)

    def tearDown(self):
        sys.path[:] = self._original_sys_path

    def test_adds_path_when_not_present(self):
        mg_root = Path('/tmp/not_actually_on_disk/MeronomyGraph_root')
        self.assertNotIn(str(mg_root), sys.path)

        SLAMData.ensure_MeronomyGraph_importable(mg_root)

        self.assertIn(str(mg_root), sys.path)
        self.assertEqual(sys.path[0], str(mg_root))

    def test_does_not_duplicate_path_when_already_present(self):
        mg_root = Path('/tmp/not_actually_on_disk/MeronomyGraph_root')
        sys.path.insert(0, str(mg_root))
        expected_count = sys.path.count(str(mg_root))

        SLAMData.ensure_MeronomyGraph_importable(mg_root)

        self.assertEqual(sys.path.count(str(mg_root)), expected_count)


if __name__ == '__main__':
    unittest.main()

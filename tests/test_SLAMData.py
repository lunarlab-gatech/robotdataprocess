import itertools
import os
from pathlib import Path
from robotdataprocess.data_types.LoopClosureData.LoopClosureData import LoopClosureData
from robotdataprocess.data_types.SLAMData import SLAMData
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


if __name__ == '__main__':
    unittest.main()

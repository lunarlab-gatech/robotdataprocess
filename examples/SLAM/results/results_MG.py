import cProfile
from dataclasses import dataclass
import getpass
import importlib.util
import sys
from pathlib import Path
from robotdataprocess import OdometryData
from types import ModuleType
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.RobotGroup import RobotGroup
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator

_EXAMPLES_DIR = Path(__file__).parent.parent.parent

def _load_results_ROMAN_module(name: str, path: Path) -> ModuleType:
    """
    Loads one dataset's results_ROMAN.py under a distinct module name, since all three share the
    filename. Registers it in ``sys.modules`` so its top-level functions can be pickled by
    reference (needed for ``multiprocessing.Pool``), not just held live in this process.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_airmuseum = _load_results_ROMAN_module("airmuseum_results_ROMAN", _EXAMPLES_DIR / "AirMuseum" / "results" / "results_ROMAN.py")
_hercules = _load_results_ROMAN_module("hercules_results_ROMAN", _EXAMPLES_DIR / "Hercules" / "results" / "results_ROMAN.py")
_kimera_multi = _load_results_ROMAN_module("kimera_multi_results_ROMAN", _EXAMPLES_DIR / "KimeraMulti" / "results" / "results_ROMAN.py")

_LOAD_GT_DATA_BY_DATASET_NAME: Dict[str, Callable[[str, List[str]], List[OdometryData]]] = {
    "airmuseum": _airmuseum.load_gt_data_ROMAN,
    "hercules": _hercules.load_gt_data_ROMAN,
    "kimera_multi": _kimera_multi.load_gt_data_ROMAN,
}

@dataclass(frozen=True)
class _LoadGtDataMG:
    """
    Picklable ``load_gt_data_fn``, dispatching to the right dataset's loader by dataset_name --
    a class rather than a closure so ``multiprocessing.Pool.starmap`` in ``run_evaluation`` can
    pickle it.

    Attributes:
        dataset_name_by_seq: Maps each dataset_seq being evaluated to its dataset_name.
    """
    dataset_name_by_seq: Dict[str, str]

    def __call__(self, dataset_seq: str, robot_names: List[str]) -> List[OdometryData]:
        dataset_name = self.dataset_name_by_seq[dataset_seq]
        return _LOAD_GT_DATA_BY_DATASET_NAME[dataset_name](dataset_seq, robot_names)

def _make_airmuseum_groups() -> List[RobotGroup]:
    """Builds the three AirMuseum all-robots-aligned groups, one per evaluated scenario."""
    viz_config = _airmuseum.make_viz_config()
    robots = ("drone", "robotA", "robotB", "robotC")
    return [RobotGroup(robots=robots, dataset_name="airmuseum", dataset_seq=seq, label=seq, viz_config=viz_config)
            for seq in ["Scenario3", "Scenario4", "Scenario5"]]

def _make_hercules_groups() -> List[RobotGroup]:
    """Builds the four HERCULES all-robots-aligned groups, one per evaluated scenario."""
    robots = ("Husky1", "Husky2", "Drone1", "Drone2")
    return [RobotGroup(robots=robots, dataset_name="hercules", dataset_seq=seq, label=seq,
                        viz_config=_hercules.make_viz_config(seq))
            for seq in ["V2.3.AP", "V2.3.AC", "V2.4.C", "V2.4.F"]]

def _make_kimera_multi_groups() -> List[RobotGroup]:
    """Builds the three Kimera-Multi all-robots-aligned groups, one per evaluated sequence."""
    viz_config = _kimera_multi.make_viz_config()
    eight_robots = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek")
    six_robots = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth")
    return [
        RobotGroup(robots=eight_robots, dataset_name="kimera_multi", dataset_seq="campus_tunnels_1207_compressed",
                   label="tunnels", viz_config=viz_config),
        RobotGroup(robots=eight_robots, dataset_name="kimera_multi", dataset_seq="campus_hybrid_1208_compressed",
                   label="hybrid", viz_config=viz_config),
        RobotGroup(robots=six_robots, dataset_name="kimera_multi", dataset_seq="campus_outdoor_1014_compressed",
                   label="outdoor", viz_config=viz_config),
    ]

def main():
    """
    Aligns all robots together for each of the ten dataset-sequence scenarios evaluated for this
    comparison (three AirMuseum, four HERCULES, three Kimera-Multi) in a single run_evaluation
    call -- unlike each dataset's own results_ROMAN.py, which instead groups only some of a
    dataset sequence's robots together at a time (e.g. pairwise, or by robot count). Requires
    AirMuseum's "ROMAN_O_SM" results directory to be renamed/aliased to "ROMAN_O" beforehand, to
    match HERCULES/Kimera-Multi's run name.

    See :meth:`SLAMEvaluator.run_evaluation` for the outputs produced.
    """
    robot_groups = _make_airmuseum_groups() + _make_hercules_groups() + _make_kimera_multi_groups()
    dataset_name_by_seq = {group.dataset_seq: group.dataset_name for group in robot_groups}

    run_names = ["ROMAN_O", "MG_TS_SM", "MG_SM"]
    user = getpass.getuser()
    figures_base_dir = Path('/home/' + user + '/Research/robotdataprocess/figures')
    roman_root = Path('/home/' + user + '/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    SLAMEvaluator.run_evaluation(roman_root, Path("AllDatasets"), run_names, robot_groups,
                             critical_invocation_params, figures_base_dir,
                             _LoadGtDataMG(dataset_name_by_seq), ate_threshold_m=20.0)

if __name__ == "__main__":
    if "--profile" in sys.argv:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        profiler.dump_stats("profile.out")
    else:
        main()

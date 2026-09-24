import cProfile
from dataclasses import dataclass
from enum import Enum
import getpass
import importlib.util
import itertools
import sys
from pathlib import Path
from robotdataprocess import OdometryData
from types import ModuleType
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from robotdataprocess.eval.RobotGroup import RobotGroup
from robotdataprocess.eval.SLAMEvaluator import SLAMEvaluator, FigureOutputLevel

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

class GroupingMode(Enum):
    """Whether to align all of a sequence's robots together, or evaluate every pair separately."""
    GLOBAL = 0
    PAIRWISE = 1

def _make_groups(dataset_name: str, seqs: List[tuple], mode: GroupingMode) -> List[RobotGroup]:
    """
    Builds one dataset's ``RobotGroup``\\ s from a list of ``(dataset_seq, seq_label, robots,
    viz_config, excluded_pairs)`` entries: one all-robots-aligned group per entry under
    ``GroupingMode.GLOBAL``, or one group per pair under ``GroupingMode.PAIRWISE`` (skipping any
    pair listed in ``excluded_pairs``, e.g. pairs with no ground-truth overlap, which would make
    evaluating them unfair -- each pair is sorted before comparing, so ``excluded_pairs`` entries
    don't need to match a particular order). ``seq_label`` is a short display name for
    ``dataset_seq`` (they may be the same string). Every label starts with
    ``"<dataset_name>_<seq_label>"``, with the pair's abbreviation appended under
    ``GroupingMode.PAIRWISE``, since sequences sharing a robot roster would otherwise collide.
    """
    groups = []
    for dataset_seq, seq_label, robots, viz_config, excluded_pairs in seqs:
        base_label = f"{dataset_name}\n{seq_label}"
        sorted_excluded_pairs = {tuple(sorted(p)) for p in excluded_pairs}
        if mode == GroupingMode.GLOBAL:
            groups.append(RobotGroup(robots=robots, dataset_name=dataset_name, dataset_seq=dataset_seq,
                                      label=base_label, viz_config=viz_config))
        else:
            for pair in itertools.combinations(sorted(robots), 2):
                if pair in sorted_excluded_pairs:
                    continue
                label = f"{base_label}\n{SLAMEvaluator.group_label(pair)}"
                groups.append(RobotGroup(robots=pair, dataset_name=dataset_name, dataset_seq=dataset_seq,
                                          label=label, viz_config=viz_config))
    return groups

def _make_airmuseum_groups(mode: GroupingMode) -> List[RobotGroup]:
    """Builds AirMuseum's robot groups per scenario: all 4 robots aligned, or all 6 pairs."""
    viz_config = _airmuseum.make_viz_config()
    robots = ("drone", "robotA", "robotB", "robotC")
    scenario3_excluded_pairs = [("robotA", "robotC")] # No Overlap
    scenario4_excluded_pairs = [("drone", "robotA")]  # No Overlap
    scenario5_excluded_pairs = [] # All overlap
    seqs = [
        ("Scenario3", "Scenario3", robots, viz_config, scenario3_excluded_pairs),
        ("Scenario4", "Scenario4", robots, viz_config, scenario4_excluded_pairs),
        ("Scenario5", "Scenario5", robots, viz_config, scenario5_excluded_pairs),
    ]
    return _make_groups("airmuseum", seqs, mode)

def _make_hercules_groups(mode: GroupingMode) -> List[RobotGroup]:
    """Builds HERCULES's robot groups per scenario: all 4 robots aligned, or all 6 pairs."""
    robots = ("Husky1", "Husky2", "Drone1", "Drone2")
    excluded_pairs = [] # All overlap
    seq_names = ["V2.3.AC", "V2.3.AP", "V2.4.C", "V2.4.F"] 
    viz_configs = [_hercules.make_viz_config(seq) for seq in seq_names]
    seqs = [(seq, seq, robots, viz_config, excluded_pairs) for seq, viz_config in zip(seq_names, viz_configs)]
    return _make_groups("hercules", seqs, mode)

def _make_kimera_multi_groups(mode: GroupingMode) -> List[RobotGroup]:
    """
    Builds Kimera-Multi's robot groups per sequence: all robots aligned (8 for tunnels/hybrid,
    6 for outdoor), or every pair (28 pairs for tunnels/hybrid, 15 for outdoor), minus pairs with
    no ground-truth overlap.
    """
    viz_config = _kimera_multi.make_viz_config()
    eight_robots = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth", "apis", "sobek")
    six_robots = ("acl_jackal", "acl_jackal2", "sparkal1", "sparkal2", "hathor", "thoth")
    tunnels_excluded_pairs = [] # All Overlap
    hybrid_excluded_pairs = [("acl_jackal", "apis"), ("acl_jackal", "sobek")]        # Very little Overlap
    outdoor_excluded_pairs = [("acl_jackal", "sparkal1"), ("acl_jackal2", "hathor"), # Very little Overlap
                              ("acl_jackal2", "thoth"),                              # No Overlap
                              ("sparkal1", "hathor"),                                # Very little Overlap
                              ("sparkal1", "thoth")]                                 # No Overlap
    seqs = [
        ("campus_tunnels_1207_compressed", "tunnels", eight_robots, viz_config, tunnels_excluded_pairs),
        ("campus_hybrid_1208_compressed", "hybrid", eight_robots, viz_config, hybrid_excluded_pairs),
        ("campus_outdoor_1014_compressed", "outdoor", six_robots, viz_config, outdoor_excluded_pairs),
    ]
    return _make_groups("kimera_multi", seqs, mode)

def main():
    """
    Evaluates dataset-sequence scenarios across AirMuseum, HERCULES, and Kimera-Multi in a
    single run_evaluation call -- unlike each dataset's own results_ROMAN.py, which instead
    groups only some of a dataset sequence's robots together at a time. Under
    ``GroupingMode.GLOBAL``, all robots of all sequences of all three datasets are aligned
    together. Under ``GroupingMode.PAIRWISE``, a single dataset and sequence is selected instead,
    with every robot pair of that sequence its own group. Requires AirMuseum's "ROMAN_O_SM"
    results directory to be renamed/aliased to "ROMAN_O" beforehand, to match
    HERCULES/Kimera-Multi's run name.

    See :meth:`SLAMEvaluator.run_evaluation` for the outputs produced.
    """
    mode = GroupingMode.PAIRWISE
    only_heterogeneous = True
    all_dir_name = "all_heterogeneous" if only_heterogeneous else "all"

    if mode == GroupingMode.GLOBAL:
        robot_groups = _make_airmuseum_groups(mode) + _make_hercules_groups(mode) + _make_kimera_multi_groups(mode)
        output_dir = Path(all_dir_name) / mode.name.lower()
    else:
        dataset_name = "airmuseum"
        dataset_seq = "Scenario5"
        group_fns_by_dataset_name: Dict[str, Callable[[GroupingMode], List[RobotGroup]]] = {
            "airmuseum": _make_airmuseum_groups,
            "hercules": _make_hercules_groups,
            "kimera_multi": _make_kimera_multi_groups,
        }
        robot_groups = [g for g in group_fns_by_dataset_name[dataset_name](mode) if g.dataset_seq == dataset_seq]
        output_dir = Path(all_dir_name) / mode.name.lower() / dataset_name / dataset_seq

    if only_heterogeneous:
        robot_groups = [g for g in robot_groups if not all("drone" in r.lower() for r in g.robots)
                                                and any("drone" in r.lower() for r in g.robots)]

    dataset_name_by_seq = {group.dataset_seq: group.dataset_name for group in robot_groups}

    run_names = ["ROMAN_O", "MG_TS_SM", "MG_SM"]
    user = getpass.getuser()
    figures_base_dir = Path('/home/' + user + '/Research/robotdataprocess/figures')
    roman_root = Path('/home/' + user + '/Research/ROMAN_DEVEL')
    critical_invocation_params = {"use_lidar": False, "use_gt_odom": False}

    SLAMEvaluator.run_evaluation(roman_root, output_dir, run_names, robot_groups,
                             critical_invocation_params, figures_base_dir,
                             _LoadGtDataMG(dataset_name_by_seq), ate_threshold_m=20.0,
                             figure_output_level=FigureOutputLevel.ESSENTIAL)

if __name__ == "__main__":
    if "--profile" in sys.argv:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        profiler.dump_stats("profile.out")
    else:
        main()

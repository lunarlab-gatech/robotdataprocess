import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'results'))
from results_AirMuseum import load_gt_data_ROMAN

def main():
    dataset_name = "Scenario5"

    gt_data_robotA, = load_gt_data_ROMAN(dataset_name, ["robotA"])
    gt_data_robotA.visualize_3D([], ["robotA"])

    gt_data_drone, = load_gt_data_ROMAN(dataset_name, ["drone"])
    gt_data_drone.visualize_3D([], ["drone"])

if __name__ == "__main__":
    main()

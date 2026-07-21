import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'results'))
from results_AirMuseum import load_gt_data_ROMAN

def main():
    dataset_name = "Scenario5"
    robot_names = ["drone", "robotA", "robotB", "robotC"]

    for rn in robot_names:
        gt_data = load_gt_data_ROMAN(dataset_name, [rn])
        gt_data.visualize_3D([], [rn], axes_length=1, axes_interval=20)

if __name__ == "__main__":
    main()

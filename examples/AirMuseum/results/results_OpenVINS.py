import getpass
from pathlib import Path
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
import sys
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from results_ROMAN import load_gt_data_ROMAN, NAME_TO_FRAME_MAP

def main():
    robot_names: List[str] = ["drone", "robotA", "robotB", "robotC"]
    dataset_seq: str = "Scenario5"
    skip_robots: List = ["robotA", "robotB", "robotC"]

    # Get paths and names
    dataset_path = Path('/media') / getpass.getuser() / 'T73' / 'AirMuseum_dataset' / dataset_seq
    results_path = Path(dataset_path) / "results"

    for robot_name in robot_names:
        if robot_name in skip_robots:
            continue
        print("\n=== Processing results for robot:", robot_name)

        # Load the OpenVINS estimate. TODO: Check if output of OpenVINS is camera frame or IMU frame.
        est_data = OdometryData.from_txt(
            results_path / 'openvins' / robot_name / 'ov_estimate.txt',
            "world", "robot", CoordinateFrame.FLU, True, [0, 5, 6, 7, 4, 1, 2, 3])
        est_data.redefine_local_axes(NAME_TO_FRAME_MAP[robot_name], CoordinateFrame.FLU)

        # Load the ground truth, already in FLU frame
        gt_data: OdometryData = load_gt_data_ROMAN(dataset_seq, [robot_name])[0]

        # Calculate RMS ATE, among other metrics
        metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1, visualize=True)
        print("\nMetrics for robot:", robot_name)
        print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)

        print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse)
        print("Robot: ", robot_name, "RMS RTE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)

if __name__ == "__main__":
    main()

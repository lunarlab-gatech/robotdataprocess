from decimal import Decimal
import getpass
import numpy as np
import os
from pathlib import Path
from robotdataprocess import TransformType, CameraData, ImageData, ImageDataOnDisk, TransformationData
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R
import sys
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from results_ROMAN import load_gt_data_ROMAN, NAME_TO_FRAME_MAP

def main():
    robot_names: List[str] = ["drone", "robotA", "robotB", "robotC"]
    dataset_seq: str = "Scenario5"
    skip_robots: List = ["drone"]

    # Define camera constants
    robot_left_cam_map: dict = {
        'drone': 'cam100',
        'robotA': 'cam101',
        'robotB': 'cam101',
        'robotC': 'cam101',
    }
    cam_id_to_calib_name: dict = {
        'cam100': 'cam0', 
        'cam101': 'cam1'
    }
    cam_id_to_bag_name_map: dict = {
        'cam100': 'cam100_imu.bag', 
        'cam101': 'cam101.bag'
    }

    # Get paths and names
    dataset_path = Path('/media') / getpass.getuser() / 'T73' / 'AirMuseum_dataset' / dataset_seq
    dataset_version: Path = Path(dataset_path).name
    dataset_config_path: Path = Path(dataset_path).parent
    base_path = Path(dataset_path) / "data"
    results_path = Path(dataset_path) / "results"

    for robot_name in robot_names:
        if robot_name in skip_robots:
            continue
        print("\n=== Processing results for robot:", robot_name)
        input_path = base_path / robot_name

        # Get camera attributes for this robot
        left_cam_id = robot_left_cam_map[robot_name]
        right_cam_id = 'cam101' if left_cam_id == 'cam100' else 'cam100'
        left_bag = input_path / cam_id_to_bag_name_map[left_cam_id]
        right_bag = input_path / cam_id_to_bag_name_map[right_cam_id]

        # ============================ Get images and intrinsics & stereo undistort/rectify =========================
        # Load stereo intrinstics
        calib_name = robot_name + "_cameras_calib.yaml"
        cam_data_left, cam_data_right = CameraData.from_kalibr_stereo(dataset_config_path / 'sensors' / calib_name, cam_id_to_calib_name[left_cam_id], cam_id_to_calib_name[right_cam_id], alpha=0.0)

        # Load the Camera images along with depth
        left_image_data = ImageDataOnDisk.from_ros1_bag(left_bag, f'/{robot_name}/{left_cam_id}/image_raw')
        right_image_data = ImageDataOnDisk.from_ros1_bag(right_bag, f'/{robot_name}/{right_cam_id}/image_raw')
        assert left_image_data.encoding == right_image_data.encoding, "Left/Right image encodings must match!"
        assert left_image_data.encoding == ImageData.ImageEncoding.Mono8, "Expected AirMuseum imagery to be Mono8"

        # Align timestamps with the IMU's timestamps
        CameraData.align_ImageData_and_CameraData_to_imu_ts([left_image_data], cam_data_left)
        CameraData.align_ImageData_and_CameraData_to_imu_ts([right_image_data], cam_data_right)

        # Get only synced images
        ImageDataOnDisk.crop_to_matched(left_image_data, right_image_data, Decimal('0.01'))

        # Undistort, saving R before it is set to identity by undistortion.
        R_rectify_left = cam_data_left.get_R()
        ImageDataOnDisk.undistort_imagery_stereo(left_image_data, right_image_data, cam_data_left, cam_data_right)

        # ==================================== Load Transformations =========================================
        # Load and calculate T_IMU_CAMERA
        H_O_to_I = TransformationData.from_kalibr(dataset_config_path / 'sensors' / calib_name, "cam0", "T_cam_imu", CoordinateFrame.FLU)
        local_axes_correction = TransformationData(
            H_O_to_I.child_frame_id, H_O_to_I.child_frame_id, np.zeros(3),
            R.from_matrix(CoordinateFrame.get_rotation(CoordinateFrame.FLU, NAME_TO_FRAME_MAP[robot_name])).as_quat(),
            H_O_to_I.frame)
        H_O_to_I = H_O_to_I.apply_transformation_right_side(local_axes_correction) # Updates local axes from default to FLU
        
        # Calculate T_CAMERA_FLU (Should be called T_CAMERA_IMU)
        H_I_to_O = H_O_to_I.invert()

        # ==================================== Load Odometry =========================================

        # Load the data
        est_data = OdometryData.from_tum(results_path / 'ORB-SLAM3' / ('AirMuseum_' + robot_name + '_ORBSLAM3_trajectory.txt'), "world", "robot", CoordinateFrame.LDB)

        # Re-target est_data from "pose of rectified camera" to "pose of IMU" via H_I_to_OR = H_I_to_O @ T_O_to_OR.
        T_O_to_OR = TransformationData(
            H_I_to_O.child_frame_id, H_I_to_O.child_frame_id + 'R', np.zeros(3),
            R.from_matrix(R_rectify_left.T).as_quat(), H_I_to_O.frame)
        H_I_to_OR = H_I_to_O.apply_transformation_right_side(T_O_to_OR)
        est_data.apply_transformation_right_side(H_I_to_OR.invert().as_matrix())

        # Convert from LDB frame to FLU frame
        est_data.to_coordinate_frame(CoordinateFrame.FLU, transform_type=TransformType.ROTATION)

        # Load gt_data
        gt_data: OdometryData = load_gt_data_ROMAN(dataset_seq, [robot_name])[0]

        # Visualize the estimated versus gt data
        est_data.visualize_3D([gt_data],["Est", "GT"], axes_length=1, axes_interval=500)
        # TODO: Verify all est_data transformations have a single axis that corresponds to down!

        # Calculate RMS ATE, among other metrics
        # metrics_dictionary, _, _ = OdometryData.align_and_calculate_traj_errors(gt_data, est_data, max_diff=0.1,
        #                                                                     visualize=False, axes_interval=2000)
        # print("\nMetrics for robot: ", robot_name)
        # print("Robot: ", robot_name, "RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
        # print("Robot: ", robot_name, "RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse, "\n")

        # print("Robot: ", robot_name, "RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)
        # print("Robot: ", robot_name, "RMS RPE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)


if __name__ == "__main__":
    main()

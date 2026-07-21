import sys
import getpass
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from robotdataprocess import TransformationData, CoordinateFrame
sys.path.insert(0, str(Path(__file__).parent.parent / 'results'))
from results_AirMuseum import NAME_TO_FRAME_MAP

def main():
    robot_names = ["drone", "robotA", "robotB", "robotC"]
    dataset_path = Path('/media/' + getpass.getuser() + '/T73/AirMuseum_dataset/')

    for robot_name in robot_names:
        # ==================================== Load Transformations =========================================
        calib_name = robot_name + "_cameras_calib.yaml"
        H_O_to_I = TransformationData.from_kalibr(dataset_path / 'sensors' / calib_name, "cam0", "T_cam_imu", CoordinateFrame.FLU)

        # Re-express the IMU side of the calibration in FLU, matching the GT trajectories
        curr_local_frame = NAME_TO_FRAME_MAP[robot_name]
        local_axes_correction = TransformationData(
            H_O_to_I.child_frame_id, H_O_to_I.child_frame_id, np.zeros(3),
            R.from_matrix(CoordinateFrame.get_rotation(CoordinateFrame.FLU, curr_local_frame)).as_quat(),
            H_O_to_I.frame)
        H_O_to_I_FLU = H_O_to_I.apply_transformation_right_side(local_axes_correction)

        H_I_to_O = H_O_to_I_FLU.invert()
        TransformationData.visualize([H_I_to_O])

if __name__ == "__main__":
    main()

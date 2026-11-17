import cv2
import getpass
import numpy as np
from pathlib import Path
from robotdataprocess import CoordinateFrame, ImageDataOnDisk, ImuData, OdometryData
from robotdataprocess.utils.conversion_utils import dec_arr_to_float_arr
import sys
import tqdm
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "results"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_ROMAN
import examples.KimeraMulti.verification.check_vio_gt_time_offset as vgo

IMAGE_STRIDE = 3  # use every 3rd color frame (~10 Hz)
IMAGE_DIFF_SIZE = (80, 60)  # (width, height) images are downscaled to before differencing

def windowed_profile(t: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates a sampled signal onto the profile grid and box-averages it over WINDOW."""
    grid: np.ndarray = vgo.make_grid(t[0], t[-1])
    kernel_len: int = int(round(vgo.WINDOW / vgo.GRID_DT))
    return grid, np.convolve(np.interp(grid, t, values), np.ones(kernel_len) / kernel_len, mode='same')

def image_change_profile(images: ImageDataOnDisk) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image-content change rate: mean absolute difference between consecutive (strided, downscaled,
    grayscale) frames divided by their time gap, stamped at the midpoint of the two header stamps.

    Args:
        images (ImageDataOnDisk): RGB image stream to profile.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Grid times (s) and windowed change rate (intensity/s).
    """
    t: np.ndarray = dec_arr_to_float_arr(np.array(images.timestamps))[::IMAGE_STRIDE]
    small = [cv2.resize(cv2.cvtColor(images.images[i], cv2.COLOR_RGB2GRAY), IMAGE_DIFF_SIZE,
                        interpolation=cv2.INTER_AREA).astype(np.float32)
             for i in tqdm.tqdm(range(0, len(images.images), IMAGE_STRIDE), desc="Differencing images")]
    rates = np.array([np.abs(b - a).mean() for a, b in zip(small[:-1], small[1:])]) / np.diff(t)
    return windowed_profile((t[:-1] + t[1:]) / 2, rates)

def main():
    """
    Determines which clock the raw-bag sensors (color images, D455 IMU, wheel odometry) of a
    Kimera-Multi robot agree with: the precomputed Kimera-VIO odometry or the ground-truth CSV.
    Each sensor's profile is cross-correlated against the GT and VIO angular-speed profiles
    (linear speed for wheel odometry); a best offset near 0 s means the sensor shares that clock.
    """
    user = getpass.getuser()
    data_dir = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data'
    dataset_seq = results_ROMAN.TUNNELS_SEQ
    robots = ["acl_jackal", "sparkal1", "sparkal2"]  # acl_jackal is a control with 0 s VIO/GT offset

    rows = []
    gt_by_robot = dict(zip(robots, results_ROMAN.load_gt_data_ROMAN(dataset_seq, robots)))
    for robot in robots:
        raw_bag: Path = data_dir / dataset_seq / ('12_07_' + robot + '.bag')
        vio = OdometryData.from_ros1_bag(data_dir / dataset_seq / 'Kimera-VIO-Odom' / (robot + '.bag'),
                                         '/' + robot + '/kimera_vio_ros/odometry', CoordinateFrame.NONE)
        wheel = OdometryData.from_ros1_bag(raw_bag, '/' + robot + '/jackal_velocity_controller/odom', CoordinateFrame.NONE)
        imu = ImuData.from_ros1_bag(raw_bag, '/' + robot + '/forward/imu', 'imu')
        images = ImageDataOnDisk.from_ros1_bag(raw_bag, '/' + robot + '/forward/color/image_raw/compressed')

        # Build reference (GT, VIO) and sensor profiles
        gt_grid, gt_lin, gt_ang = vgo.motion_profiles(gt_by_robot[robot])
        vio_grid, vio_lin, vio_ang = vgo.motion_profiles(vio)
        wheel_grid, wheel_lin, _ = vgo.motion_profiles(wheel)
        sensors: Dict[str, Tuple[np.ndarray, np.ndarray, bool]] = {
            'wheel': (wheel_grid, wheel_lin, True),
            'imu': (*windowed_profile(dec_arr_to_float_arr(imu.timestamps),
                                      np.linalg.norm(dec_arr_to_float_arr(imu.ang_vel), axis=1)), False),
            'image': (*image_change_profile(images), False),
        }
        images.images.close()

        # Correlate each sensor against GT and VIO (offset: reference time = sensor time + offset)
        rows.append(f"{robot:12s} img0-gt0 {float(images.timestamps[0] - gt_by_robot[robot].timestamps[0]):+.2f}  "
                    f"img0-vio0 {float(images.timestamps[0] - vio.timestamps[0]):+.2f}")
        for name, (grid, profile, use_lin) in sensors.items():
            gt_ref, vio_ref = (gt_lin, vio_lin) if use_lin else (gt_ang, vio_ang)
            rows.append(f"    {name:6s} vs GT: {vgo.summarize(vgo.cross_correlate(grid, profile, gt_grid, gt_ref))} | "
                        f"vs VIO: {vgo.summarize(vgo.cross_correlate(grid, profile, vio_grid, vio_ref))}")

    print("Columns per reference: corr at 0 s, best corr, best offset (reference time = sensor time + offset)")
    print('\n'.join(rows))


if __name__ == "__main__":
    main()

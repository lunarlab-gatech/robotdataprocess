from decimal import Decimal
from getpass import getuser
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from robotdataprocess import CameraData, ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData

# Which physical camera (cam100 or cam101) is the *left* eye of the stereo
# rig, per robot. cam0/cam1 in each robot's calibration YAML always
# correspond to cam100/cam101 respectively.
ROBOT_LEFT_CAM = {
    'drone': 'cam100',
    'robotA': 'cam101',
    'robotB': 'cam101',
    'robotC': 'cam101',
}
CAM_ID_TO_CALIB_NAME = {'cam100': 'cam0', 'cam101': 'cam1'}
CAM_ID_TO_BAG_NAME = {'cam100': 'cam100_imu.bag', 'cam101': 'cam101.bag'}

def main():

    scenario: str = "Scenario5"
    robot_name: str = "drone"
    min_depth: float = 0.1
    max_depth: float = 20.0
    sync_tolerance: Decimal = Decimal('0.01')

    depth_method_name = "FoundationStereo"
    dataset_root: Path = Path('/media') / getuser() / 'T73' / 'AirMuseum_dataset'
    depth_folder: Path = dataset_root / scenario / 'results' / depth_method_name / robot_name / 'depth'
    depth_data = ImageDataOnDisk.from_npy_files(depth_folder, f'{robot_name}/{depth_method_name}')

    left_cam_id = ROBOT_LEFT_CAM[robot_name]
    left_bag = dataset_root / scenario / 'data' / robot_name / CAM_ID_TO_BAG_NAME[left_cam_id]
    left_topic = f'/{robot_name}/{left_cam_id}/image_raw'
    calib_path = dataset_root / 'sensors' / f'{robot_name}_cameras_calib.yaml'

    image_data = ImageDataOnDisk.from_ros1_bag(left_bag, left_topic)
    cam = CameraData.from_kalibr_mono(calib_path, CAM_ID_TO_CALIB_NAME[left_cam_id])
    CameraData.align_ImageData_and_CameraData_to_imu_ts([image_data], cam)

    # Depth timestamps are already on the IMU clock (see
    # FoundationStereo's run_inference_robotdataprocess.py), so crop both
    # sequences down to their matched (within-tolerance) frames
    ImageDataOnDisk.crop_to_matched(image_data, depth_data, sync_tolerance)

    fig, ax = plt.subplots()
    im = ax.imshow(_to_vis(image_data, depth_data, 0, min_depth, max_depth))
    ax.set_title(f"Frame 0/{len(depth_data.images)}")

    for i in range(len(depth_data.images)):
        im.set_data(_to_vis(image_data, depth_data, i, min_depth, max_depth))
        ax.set_title(f"Frame {i}/{len(depth_data.images)}")
        plt.pause(0.001)

    plt.show()

def _to_rgb(image_data: ImageDataOnDisk, i: int) -> np.ndarray:
    image = image_data.images[i]
    if image_data.encoding == ImageData.ImageEncoding.BGR8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_data.encoding == ImageData.ImageEncoding.Mono8:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def _to_vis(image_data: ImageDataOnDisk, depth_data: ImageDataOnDisk, i: int,
           min_depth: float, max_depth: float) -> np.ndarray:
    depth = depth_data.images[i]
    normalized = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1) * 255

    # Invert so near depth renders as the "hot"/red end of the colormap,
    # matching FoundationStereo's run_inference_robotdataprocess.py convention.
    depth_vis = cv2.applyColorMap((255 - normalized).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

    return np.concatenate([_to_rgb(image_data, i), depth_vis], axis=1)

if __name__ == "__main__":
    main()

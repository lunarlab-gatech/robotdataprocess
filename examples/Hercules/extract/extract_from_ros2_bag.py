import argparse
from decimal import Decimal
from pathlib import Path
import numpy as np
from PIL import Image
import tqdm
from typing import Union

from robotdataprocess import ImuData, OdometryData, CoordinateFrame, LiDARData
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory

# =============================================================================
# Edit topic names here — set any topic to None to skip that sensor
# =============================================================================
IMU_TOPIC        = '/gnss/imu'
ODOM_TOPIC       = '/gnss/ground_truth'
LEFT_IMG_TOPIC   = '/camera_left/image_raw'
RIGHT_IMG_TOPIC  = '/camera_right/image_raw'
LIDAR_TOPIC      = '/velodyne/points'

IMU_TOPIC        = None
ODOM_TOPIC       = None
LEFT_IMG_TOPIC   = None
RIGHT_IMG_TOPIC  = None
LIDAR_TOPIC      = '/velodyne/points'

# =============================================================================


def _save_lidar_to_npy_files(lidar_data: LiDARData, output_dir: Path):
    """
    Saves each LiDAR scan as an individual .npy file named by its timestamp.
    Files contain (N, 3) float32 arrays, or (N, 4) if ring channels are present.
    This format is directly loadable via LiDARData.from_npy_files().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    has_channels = lidar_data.channels is not None

    pbar = tqdm.tqdm(total=lidar_data.len(), desc="Saving LiDAR .npy files...", unit=" scans")
    for i in range(lidar_data.len()):
        ts = lidar_data.timestamps[i]
        pts = lidar_data.point_clouds[i].astype(np.float32)

        if has_channels:
            chans = lidar_data.channels[i].astype(np.uint16).reshape(-1, 1).astype(np.float32)
            data = np.concatenate([pts, chans], axis=1)  # (N, 4)
        else:
            data = pts  # (N, 3)

        np.save(str(output_dir / f"{ts}.npy"), data)
        pbar.update()
    pbar.close()


def _save_imu_to_txt(imu_data: ImuData, output_path: Path):
    """
    Saves IMU data to a space-separated .txt file compatible with ImuData.from_txt_file().
    Format per line: timestamp acc_x acc_y acc_z gyro_x gyro_y gyro_z [qx qy qz qw]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_orientations = imu_data.orientations is not None

    pbar = tqdm.tqdm(total=imu_data.len(), desc="Saving IMU .txt...", unit=" msgs")
    with open(str(output_path), 'w') as f:
        for i in range(imu_data.len()):
            ts  = imu_data.timestamps[i]
            acc = imu_data.lin_acc[i]
            gyr = imu_data.ang_vel[i]
            line = f"{ts} {acc[0]} {acc[1]} {acc[2]} {gyr[0]} {gyr[1]} {gyr[2]}"
            if has_orientations:
                ori = imu_data.orientations[i]  # xyzw
                line += f" {ori[0]} {ori[1]} {ori[2]} {ori[3]}"
            f.write(line + "\n")
            pbar.update()
    pbar.close()


def _save_images_to_png(img_data: ImageDataInMemory, output_dir: Path):
    """
    Saves each image as a PNG file named by its timestamp.
    Supports Mono8 and RGB8 encodings.
    For 32FC1 (depth), falls back to saving via to_npy() into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    enc = img_data.encoding

    if enc == ImageData.ImageEncoding.Mono8:
        # Grayscale PNGs
        pbar = tqdm.tqdm(total=img_data.len(), desc="Saving Mono8 PNGs...", unit=" imgs")
        for i in range(img_data.len()):
            ts = img_data.timestamps[i]
            filename = f"{ts:.9f}.png"
            pil_img = Image.fromarray(img_data.images[i], mode="L")
            pil_img.save(str(output_dir / filename), format="PNG", compress_level=1)
            pbar.update()
        pbar.close()

    elif enc == ImageData.ImageEncoding.RGB8:
        # Colour PNGs
        pbar = tqdm.tqdm(total=img_data.len(), desc="Saving RGB8 PNGs...", unit=" imgs")
        for i in range(img_data.len()):
            ts = img_data.timestamps[i]
            filename = f"{ts:.9f}.png"
            pil_img = Image.fromarray(img_data.images[i], mode="RGB")
            pil_img.save(str(output_dir / filename), format="PNG", compress_level=1)
            pbar.update()
        pbar.close()

    else:
        # For 32FC1 or other encodings, fall back to .npy bundle
        print(f"  Encoding {enc} not supported for PNG export; saving as .npy bundle instead.")
        img_data.to_npy(output_dir)


def extract_data(bag_path: str, output_dir: str):
    bag = Path(bag_path)
    out = Path(output_dir)

    # ------------------------------------------------------------------
    # IMU
    # ------------------------------------------------------------------
    if IMU_TOPIC is not None:
        print(f"\n[1/5] Extracting IMU from '{IMU_TOPIC}'...")
        imu_data = ImuData.from_ros2_bag(bag, IMU_TOPIC, 'imu')
        print(f"      Loaded {imu_data.len()} IMU measurements.")
        _save_imu_to_txt(imu_data, out / 'imu' / 'imu.txt')
        print(f"      Saved -> {out / 'imu' / 'imu.txt'}")
    else:
        print("[1/5] Skipping IMU (topic is None).")

    # ------------------------------------------------------------------
    # Odometry
    # ------------------------------------------------------------------
    if ODOM_TOPIC is not None:
        print(f"\n[2/5] Extracting Odometry from '{ODOM_TOPIC}'...")
        odom_data = OdometryData.from_ros2_bag(bag, ODOM_TOPIC, CoordinateFrame.NED)
        print(f"      Loaded {odom_data.len()} poses.")
        odom_csv = out / 'odom' / 'odom.csv'
        odom_csv.parent.mkdir(parents=True, exist_ok=True)
        odom_data.to_csv(odom_csv)
        print(f"      Saved -> {odom_csv}")
    else:
        print("[2/5] Skipping Odometry (topic is None).")

    # ------------------------------------------------------------------
    # Left camera
    # ------------------------------------------------------------------
    if LEFT_IMG_TOPIC is not None:
        print(f"\n[3/5] Extracting left camera from '{LEFT_IMG_TOPIC}'...")
        left_tmp = out / 'camera_left' / '_tmp_npy'
        left_img = ImageDataInMemory.from_ros2_bag(bag, LEFT_IMG_TOPIC, left_tmp)
        print(f"      Loaded {left_img.len()} images ({left_img.encoding}).")
        _save_images_to_png(left_img, out / 'camera_left' / 'images')
        print(f"      Saved -> {out / 'camera_left' / 'images'}/")
    else:
        print("[3/5] Skipping left camera (topic is None).")

    # ------------------------------------------------------------------
    # Right camera
    # ------------------------------------------------------------------
    if RIGHT_IMG_TOPIC is not None:
        print(f"\n[4/5] Extracting right camera from '{RIGHT_IMG_TOPIC}'...")
        right_tmp = out / 'camera_right' / '_tmp_npy'
        right_img = ImageDataInMemory.from_ros2_bag(bag, RIGHT_IMG_TOPIC, right_tmp)
        print(f"      Loaded {right_img.len()} images ({right_img.encoding}).")
        _save_images_to_png(right_img, out / 'camera_right' / 'images')
        print(f"      Saved -> {out / 'camera_right' / 'images'}/")
    else:
        print("[4/5] Skipping right camera (topic is None).")

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    if LIDAR_TOPIC is not None:
        print(f"\n[5/5] Extracting LiDAR from '{LIDAR_TOPIC}'...")
        lidar_data = LiDARData.from_ros2_bag(bag, LIDAR_TOPIC,
                                              'lidar', CoordinateFrame.NED)
        print(f"      Loaded {lidar_data.len()} scans.")
        _save_lidar_to_npy_files(lidar_data, out / 'lidar')
        print(f"      Saved -> {out / 'lidar'}/")
    else:
        print("[5/5] Skipping LiDAR (topic is None).")

    print(f"\nDone. All data written to: {out}")


def main(bag_path: str, output_dir: str):
    extract_data(bag_path=bag_path, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract sensor data from a ROS2 bag and save to files "
                    "(.npy for LiDAR, .png for images, .csv for odometry, .txt for IMU)."
    )
    parser.add_argument('--bag_path',    type=str, required=True,
                        help="Path to the ROS2 bag directory.")
    parser.add_argument('--output_dir',  type=str, required=True,
                        help="Root output directory; subdirectories are created per sensor.")
    args = parser.parse_args()

    main(bag_path=args.bag_path, output_dir=args.output_dir)

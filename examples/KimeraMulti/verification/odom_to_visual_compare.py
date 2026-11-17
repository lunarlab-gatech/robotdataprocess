import re
import getpass
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from decimal import Decimal
from pathlib import Path

from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from robotdataprocess.data_types.ImageData.ImageDataOnDisk import ImageDataOnDisk


def nearest_odom_idx(odom_ts_float: np.ndarray, query_f: float) -> int:
    return int(np.argmin(np.abs(odom_ts_float - query_f)))


def main():
    robot_name_text = "acl_jackal"
    dataset_version = "campus_outdoor_1014_compressed"
    dataset_number = re.search(r'\d+', dataset_version).group()  # e.g. "1014"
    robot_name = dataset_number[:2] + '_' + dataset_number[2:] + '_' + robot_name_text  # e.g. "10_14_acl_jackal"
    
    user = getpass.getuser()
    dataset_folder = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset'

    odom = OdometryData.from_ros1_bag(dataset_folder / 'data' / dataset_version / 'Kimera-VIO-Odom' / (robot_name_text + '.bag'), '/' + robot_name_text + '/kimera_vio_ros/odometry', CoordinateFrame.NONE)
    imgs = ImageDataOnDisk.from_ros1_bag(dataset_folder / 'data' / dataset_version / (robot_name + '.bag'), 
                                         '/' + robot_name_text + '/forward/color/image_raw/compressed')

    print("odom first timestamp:", odom.timestamps[0])
    print("imgs first timestamp:", imgs.timestamps[0])

    # Shift odom timestamps so first odom aligns with first image
    odom_ts_float = np.array([float(t) for t in odom.timestamps])
    #odom_ts_float -= odom_ts_float[0] - float(imgs.timestamps[0])

    positions = odom.positions.astype(np.float64)

    fig = plt.figure(figsize=(14, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_traj = fig.add_subplot(1, 2, 2, projection='3d')
    fig.tight_layout(pad=3)
    plt.ion()
    plt.show()

    ax_traj.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='lightgray', linewidth=0.8)
    traj_line, = ax_traj.plot([], [], [], color='steelblue', linewidth=1.5)
    cur_dot, = ax_traj.plot([], [], [], 'ro', markersize=6)
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_zlabel('Z (m)')

    x_c = (positions[:, 0].max() + positions[:, 0].min()) / 2
    y_c = (positions[:, 1].max() + positions[:, 1].min()) / 2
    z_c = (positions[:, 2].max() + positions[:, 2].min()) / 2
    half_range = max(positions[:, 0].max() - positions[:, 0].min(),
                     positions[:, 1].max() - positions[:, 1].min(),
                     positions[:, 2].max() - positions[:, 2].min()) / 2
    ax_traj.set_xlim(x_c - half_range, x_c + half_range)
    ax_traj.set_ylim(y_c - half_range, y_c + half_range)
    ax_traj.set_zlim(z_c - half_range, z_c + half_range)

    img_handle = None

    for frame_idx in range(0, len(imgs.timestamps), 50):
        img_ts_f = float(imgs.timestamps[frame_idx])
        odom_idx = nearest_odom_idx(odom_ts_float, img_ts_f)
        dt_ms = abs(img_ts_f - odom_ts_float[odom_idx]) * 1e3

        img = imgs.images[frame_idx]
        if img_handle is None:
            img_handle = ax_img.imshow(img, aspect='auto')
            ax_img.axis('off')
        else:
            img_handle.set_data(img)
        ax_img.set_title(f'Frame {frame_idx}  ts={img_ts_f:.3f}s  |  odom idx={odom_idx}  dt={dt_ms:.1f}ms')

        traj_line.set_data(positions[:odom_idx + 1, 0], positions[:odom_idx + 1, 1])
        traj_line.set_3d_properties(positions[:odom_idx + 1, 2])
        cur_dot.set_data([positions[odom_idx, 0]], [positions[odom_idx, 1]])
        cur_dot.set_3d_properties([positions[odom_idx, 2]])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

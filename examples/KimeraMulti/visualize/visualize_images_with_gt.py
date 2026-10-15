import getpass
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from robotdataprocess import ImageDataOnDisk
from robotdataprocess.utils.conversion_utils import dec_arr_to_float_arr
from scipy.spatial.transform import Rotation as R
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "results"))
import results_ROMAN


def nearest_pose_idx(pose_ts: np.ndarray, query_ts: float) -> int:
    """Index of the pose whose timestamp is closest to ``query_ts``."""
    return int(np.argmin(np.abs(pose_ts - query_ts)))


def main():
    """
    Steps through a Kimera-Multi robot's forward color images, drawing next to each one the
    ground-truth pose with the closest timestamp (top-down position plus heading arrow). On
    robot-sequences whose camera clock is ahead of the GT clock (tunnels sparkal1 ~4.4 s,
    sparkal2 ~1.0 s), the GT pose turns/moves before the image does. Close the figure to stop.
    """
    robot = "sparkal1"
    dataset_seq = results_ROMAN.TUNNELS_SEQ
    frame_stride = 10  # images are ~30 Hz, so this updates every ~0.33 s of data
    heading_len = 4.0  # meters

    user = getpass.getuser()
    dataset_number = dataset_seq.split('_')[2][:4]  # e.g. "1207"
    raw_bag = Path('/media') / user / 'T73' / 'Kimera-Multi_Dataset' / 'data' / dataset_seq / \
              (dataset_number[:2] + '_' + dataset_number[2:] + '_' + robot + '.bag')

    gt = results_ROMAN.load_gt_data_ROMAN(dataset_seq, [robot])[0]
    imgs = ImageDataOnDisk.from_ros1_bag(raw_bag, '/' + robot + '/forward/color/image_raw/compressed')
    gt_ts: np.ndarray = dec_arr_to_float_arr(gt.timestamps)
    positions: np.ndarray = dec_arr_to_float_arr(gt.positions)
    headings: np.ndarray = R.from_quat(dec_arr_to_float_arr(gt.orientations)).apply([1.0, 0.0, 0.0])  # FLU x = forward
    print(f"First image ts - first GT ts: {float(imgs.timestamps[0] - gt.timestamps[0]):+.3f} s")

    fig = plt.figure(figsize=(16, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_traj = fig.add_subplot(1, 2, 2)
    fig.tight_layout(pad=3)
    plt.ion()
    plt.show()

    ax_traj.plot(positions[:, 0], positions[:, 1], color='lightgray', linewidth=0.8)
    traj_line, = ax_traj.plot([], [], color='steelblue', linewidth=1.5)
    cur_dot, = ax_traj.plot([], [], 'ro', markersize=6)
    heading_line, = ax_traj.plot([], [], 'r-', linewidth=2)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title(f'{robot} GT (FLU, top-down)')

    img_handle = None

    for frame_idx in range(0, len(imgs.timestamps), frame_stride):
        img_ts = float(imgs.timestamps[frame_idx])
        gt_idx = nearest_pose_idx(gt_ts, img_ts)
        dt_ms = (gt_ts[gt_idx] - img_ts) * 1e3

        img = imgs.images[frame_idx]
        if img_handle is None:
            img_handle = ax_img.imshow(img, aspect='auto')
            ax_img.axis('off')
        else:
            img_handle.set_data(img)
        ax_img.set_title(f'Frame {frame_idx}  img ts={img_ts:.3f}s  |  GT idx={gt_idx}  GT-img dt={dt_ms:+.1f}ms')

        traj_line.set_data(positions[:gt_idx + 1, 0], positions[:gt_idx + 1, 1])
        cur_dot.set_data([positions[gt_idx, 0]], [positions[gt_idx, 1]])
        heading_tip = positions[gt_idx, :2] + heading_len * headings[gt_idx, :2] / max(np.linalg.norm(headings[gt_idx, :2]), 1e-9)
        heading_line.set_data([positions[gt_idx, 0], heading_tip[0]], [positions[gt_idx, 1], heading_tip[1]])

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

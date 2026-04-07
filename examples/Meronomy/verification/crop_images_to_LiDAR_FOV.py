import getpass
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from robotdataprocess import CameraData, ImageDataOnDisk, CoordinateFrame, LiDARData, TransformationData, TransformType

def project_lidar_onto_image(pts_cam: np.ndarray, K: np.ndarray, width: int, height: int):
    """
    Project (N,3) points already in camera optical frame onto the image plane.
    Returns (u, v, depth) arrays for points that fall within [0,width) x [0,height).
    """
    in_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[in_front]

    u = K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]

    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_image], v[in_image], pts_cam[in_image, 2]


def camera_crop(input_dir: str, robot_name: str):

    lidar_FOV = (-20, 20)

    # Extract RGB and LiDAR Data
    input_path = Path(input_dir).absolute()
    left_image_data = ImageDataOnDisk.from_image_files(input_path / robot_name / 'rgb_stereo_left', 'camera_color_optical_frame')
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", "lidar_link", CoordinateFrame.NED)
    lidar_data.calculate_point_channels(16, *lidar_FOV)
    lidar_data.make_dense()
    lidar_data.to_FLU_frame()

    # Load and calculate T_ODOMBASE_CAMERA
    H_R_to_C = TransformationData.from_HERCULES_settings_json(str(input_path / 'settings.json'), robot_name, "camera", "stereo_left")
    H_C_to_O = TransformationData.optical_wrt_camera(CoordinateFrame.NED, "stereo_left", "stereo_left_optical")
    H_R_to_O = H_R_to_C.apply_transformation_right_side(H_C_to_O)
    H_R_to_O_in_FLU = H_R_to_O.to_coordinate_frame(CoordinateFrame.FLU, TransformType.ROTATION) # We want optical axes preserved
    TransformationData.visualize([H_R_to_O_in_FLU])

    # Load and calculate T_Base_LIDAR
    H_R_to_L = TransformationData.from_HERCULES_settings_json(str(input_path / 'settings.json'), robot_name, "sensor", "LidarSensor1")
    H_R_to_L_in_FLU = H_R_to_L.to_coordinate_frame(CoordinateFrame.FLU, TransformType.CHANGE_OF_BASIS)
    TransformationData.visualize([H_R_to_L_in_FLU])

    # Calculate H_L_to_O
    H_L_to_O = H_R_to_L_in_FLU.invert().apply_transformation_right_side(H_R_to_O_in_FLU)
    TransformationData.visualize([H_L_to_O])

    # Define Camera parameters
    left_camera_data = CameraData.from_user_mono('/cam0', 752, 480, 376, 376, 376, 240, CameraData.DistortionModel.RADIAL_TANGENTIAL)
    left_camera_data.visualize_FOV(lidar_v_fov=lidar_FOV)

    # --- Debug: visualize points at each stage of the frame transformation ---
    pts_raw, _ = lidar_data.get_point_cloud_at_index(0)   # already FLU after to_FLU_frame()
    T = H_L_to_O.invert().as_matrix()
    pts_cam = (T[:3, :3] @ pts_raw.T + T[:3, 3:]).T       # (N, 3) in camera optical frame

    def scatter3(ax3d, pts, title, xlabel='X', ylabel='Y', zlabel='Z'):
        sc = ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], s=0.5, cmap='viridis')
        ax3d.set_title(title)
        ax3d.set_xlabel(xlabel)
        ax3d.set_ylabel(ylabel)
        ax3d.set_zlabel(zlabel)
        half = np.max(np.abs(pts)) * 0.1
        ax3d.set_xlim(-half, half)
        ax3d.set_ylim(-half, half)
        ax3d.set_zlim(-half, half)
        ax3d.set_box_aspect([1, 1, 1])
        return sc

    fig_dbg = plt.figure(figsize=(18, 6))
    fig_dbg.suptitle('LiDAR points — frame transformation debug', fontsize=13)

    ax1 = fig_dbg.add_subplot(131, projection='3d')
    scatter3(ax1, pts_raw, 'LiDAR frame (FLU)\nX=fwd, Y=left, Z=up')

    ax2 = fig_dbg.add_subplot(132, projection='3d')
    scatter3(ax2, pts_cam, 'Camera optical frame\nX=right, Y=down, Z=fwd')

    # Also show the XY projection in camera frame (what the image "sees")
    ax3 = fig_dbg.add_subplot(133)
    in_front = pts_cam[:, 2] > 0
    ax3.scatter(pts_cam[in_front, 0], pts_cam[in_front, 1],
                c=pts_cam[in_front, 2], s=0.5, cmap='viridis')
    ax3.set_title('Camera optical frame\nXY plane (Z>0 only)')
    ax3.set_xlabel('X (right)')
    ax3.set_ylabel('Y (down)')
    half_2d = np.max(np.abs(pts_cam[in_front, :2])) * 1.05
    ax3.set_xlim(-half_2d, half_2d)
    ax3.set_ylim(-half_2d, half_2d)
    ax3.invert_yaxis()
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Frame 0 — before/after crop with LiDAR projection', fontsize=13)

    # --- Before crop ---
    axes[0, 0].imshow(left_image_data.images[0])
    axes[0, 0].set_title('Image — before crop')
    axes[0, 0].axis('off')

    u, v, depth = project_lidar_onto_image(pts_cam, left_camera_data.K,
                                            left_camera_data.width, left_camera_data.height)
    axes[1, 0].imshow(left_image_data.images[0])
    sc0 = axes[1, 0].scatter(u, v, c=depth, s=1.5, cmap='viridis', alpha=0.7)
    plt.colorbar(sc0, ax=axes[1, 0], label='Depth (m)')
    axes[1, 0].set_title('LiDAR projection — before crop')
    axes[1, 0].axis('off')

    # --- Crop ---
    left_image_data.crop_images_to_LiDAR_FOV(lidar_FOV, left_camera_data)
    left_camera_data.visualize_FOV(lidar_v_fov=lidar_FOV)

    # --- After crop ---
    axes[0, 1].imshow(left_image_data.images[0])
    axes[0, 1].set_title('Image — after crop')
    axes[0, 1].axis('off')

    u, v, depth = project_lidar_onto_image(pts_cam, left_camera_data.K,
                                            left_camera_data.width, left_camera_data.height)
    axes[1, 1].imshow(left_image_data.images[0])
    sc1 = axes[1, 1].scatter(u, v, c=depth, s=1.5, cmap='viridis', alpha=0.7)
    plt.colorbar(sc1, ax=axes[1, 1], label='Depth (m)')
    axes[1, 1].set_title('LiDAR projection — after crop')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def main(dataset_num: str, robot_name: str):
    user = getpass.getuser()
    camera_crop(input_dir='/media/' + user + '/T73/Meronomy_datasets/' + dataset_num + '/data', robot_name=robot_name)

if __name__ == "__main__":
    main(dataset_num="V1.0", robot_name="Husky1")

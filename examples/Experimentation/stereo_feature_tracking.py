"""
Example script demonstrating stereo feature tracking on the HERCULES dataset.

This script loads left/right stereo images and ground truth pose from the
HERCULES dataset, performs ORB feature detection and matching, triangulates
3D points, tracks them temporally, and validates matches using GT pose
via reprojection error.
"""

from decimal import Decimal
import getpass
import numpy as np
from pathlib import Path
from typing import Union

from robotdataprocess import ImageDataOnDisk, OdometryData, CoordinateFrame
from robotdataprocess.data_types.ImageData.ImageData import stereo_feature_tracking


def run_stereo_tracking(
    input_dir: str,
    robot_name: str,
    crop_data: bool = False,
    end_time: Union[Decimal, None] = None,
    max_features: int = 1000,
    tracker_type: str = "orb",
    verbose: bool = False,
    visualize: bool = False
):
    """
    Run stereo feature tracking on HERCULES dataset.

    Args:
        input_dir: Path to HERCULES dataset data directory.
        robot_name: Name of the robot (e.g., "Husky1", "Drone1").
        crop_data: Whether to crop data to a specific time range.
        end_time: End time for cropping (required if crop_data is True).
        max_features: Maximum features to detect per image.
        tracker_type: "orb" for ORB descriptor matching, "klt" for KLT with Shi-Tomasi corners.
        verbose: If True, print per-frame results during processing.
        visualize: If True, show real-time visualization of matches.
    """
    if crop_data and end_time is None:
        raise ValueError("end_time required if crop_data is True!")

    input_path = Path(input_dir).absolute()

    print(f"Loading data for robot: {robot_name}")

    # Load left stereo images
    print("  Loading left stereo images...")
    left_images = ImageDataOnDisk.from_image_files(
        input_path / robot_name / 'rgb_stereo_left',
        'front_center_Scene_left'
    )

    # Load right stereo images
    print("  Loading right stereo images...")
    right_images = ImageDataOnDisk.from_image_files(
        input_path / robot_name / 'rgb_stereo_right',
        'front_center_Scene_right'
    )

    # Load ground truth pose
    print("  Loading ground truth pose...")
    gt_pose = OdometryData.from_txt(
        input_path / robot_name / 'pose_world_frame.txt',
        'world',
        robot_name + '/base_link',
        CoordinateFrame.NED,
        False
    )
    # Convert to FLU frame for consistency
    gt_pose.to_FLU_frame()

    # Crop data if requested
    if crop_data:
        print(f"  Cropping data to [0.0, {end_time}]...")
        left_images.crop_data(Decimal('0.0'), end_time)
        right_images.crop_data(Decimal('0.0'), end_time)
        gt_pose.crop_data(Decimal('0.0'), end_time)

    # =========================================================================
    # Stereo calibration parameters for HERCULES dataset
    # Adjust these values based on your actual camera calibration
    # =========================================================================

    image_width = left_images.width
    image_height = left_images.height
    fov_horizontal_deg = 90.0  # Typical AirSim default FOV

    # Calculate focal length from FOV: fx = (width / 2) / tan(fov / 2)
    fx = (image_width / 2) / np.tan(np.radians(fov_horizontal_deg / 2))
    fy = fx  # Assuming square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Left camera intrinsics (K1)
    K1 = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Right camera intrinsics (K2) - same as left for symmetric stereo
    K2 = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Distortion coefficients (AirSim typically has no distortion)
    D1 = np.zeros(5, dtype=np.float64)
    D2 = np.zeros(5, dtype=np.float64)

    # Rotation matrix from left to right camera (identity for parallel cameras)
    R = np.eye(3, dtype=np.float64)

    # Translation vector from left to right camera (baseline along x-axis)
    baseline = 0.055 + 0.055  # Total baseline in meters
    T = np.array([-baseline, 0, 0], dtype=np.float64)

    print(f"\nStereo calibration parameters:")
    print(f"  Image size: {image_width} x {image_height}")
    print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Baseline: {baseline} m")

    print(f"\nDataset info:")
    print(f"  Number of image frames: {left_images.len()}")
    print(f"  Number of GT pose samples: {gt_pose.len()}")
    print(f"  Left image encoding: {left_images.encoding}")
    print(f"  Right image encoding: {right_images.encoding}")

    # Run stereo feature tracking (with GT pose validation)
    print(f"\nRunning stereo feature tracking with tracker={tracker_type}, max_features={max_features}...")
    results = stereo_feature_tracking(
        left_images=left_images,
        right_images=right_images,
        gt_pose=gt_pose,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        max_features=max_features,
        match_ratio_threshold=0.75,
        tracker_type=tracker_type,
        verbose=verbose,
        visualize=visualize
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("STEREO FEATURE TRACKING RESULTS")
    print("=" * 60)
    print(f"Total frames processed: {results['num_frames']}")
    print(f"Total features tracked: {results['total_features_tracked']}")

    if results['mean_reprojection_error'] is not None:
        print(f"\nReprojection Error Statistics:")
        print(f"  Mean error: {results['mean_reprojection_error']:.2f} pixels")
        print(f"  Median error: {results['median_reprojection_error']:.2f} pixels")
        print(f"  Std deviation: {results['std_reprojection_error']:.2f} pixels")
    else:
        print("\nNo valid reprojection errors could be computed.")

    # Print per-frame summary (first 10 frames)
    print("\nPer-frame summary (first 10 frames):")
    print("-" * 60)
    for i, frame_result in enumerate(results['per_frame_results'][:10]):
        ts = frame_result['timestamp']
        n_stereo = frame_result['num_stereo_matches']
        n_tracked = frame_result['num_tracked']
        mean_err = frame_result.get('mean_reprojection_error')
        if mean_err is not None:
            print(f"  Frame {i:3d} | t={ts:.3f}s | stereo={n_stereo:3d} | tracked={n_tracked:3d} | reproj_err={mean_err:.2f}px")
        else:
            print(f"  Frame {i:3d} | t={ts:.3f}s | stereo={n_stereo:3d} | tracked={n_tracked:3d} | reproj_err=N/A")

    return results


def main():

    dataset_num = 'V2.3.C'
    robot_name = "Drone1"
    crop_end_time = None
    tracker_type = "klt"  # "orb" for ORB descriptor matching, "klt" for KLT with Shi-Tomasi corners
    verbose = True  # Set to True to see per-frame results during processing
    visualize = True  # Set to True to see real-time visualization (press 'q' to quit early)
    user = getpass.getuser()
    input_dir = f'/media/{user}/T73/Hercules_datasets/{dataset_num}/data'

    # Determine crop settings
    crop_data = crop_end_time is not None
    end_time = Decimal(str(crop_end_time)) if crop_data else None

    # Run stereo tracking
    results = run_stereo_tracking(
        input_dir=input_dir,
        robot_name=robot_name,
        crop_data=crop_data,
        end_time=end_time,
        max_features=200,
        tracker_type=tracker_type,
        verbose=verbose,
        visualize=visualize
    )

    print("\nDone!")


if __name__ == "__main__":
    main()

import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from robotdataprocess import ImageDataOnDisk

# Set bigger font sizes for plots
plt.rcParams.update({
    'font.size':24,
    'axes.titlesize':26,
    'axes.labelsize':24,
    'xtick.labelsize':22,
    'ytick.labelsize':22,
    'legend.fontsize':22,
})


def main():
    dataset_num = "V2.3.AC"
    user = getpass.getuser()
    robot_name = "Drone1"
    input_path = Path('/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data')

    # Load GT depth data
    depth_data = ImageDataOnDisk.from_npy_files(input_path / robot_name / 'depth', 'front_center_DepthPerspective')

    # Load estimated depths from CSV
    # CSV columns: timestamp, feature_id, cam_id, u, v, depth
    csv_path = Path('/home/dbutterfield3/Research/ros_workspaces/open_vins_ws/src/open_vins/feature_depths_' + robot_name + '.csv')
    estimated_df = pd.read_csv(csv_path)
    
    numeric_cols = ['timestamp', 'feature_id', 'cam_id', 'u', 'v', 'depth']
    estimated_df[numeric_cols] = estimated_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Filter for left camera only (cam_id == 0)
    estimated_df = estimated_df[estimated_df['cam_id'] == 0].copy()

    # Convert GT timestamps to float for matching
    gt_timestamps = np.array([float(t) for t in depth_data.timestamps])

    # Storage for errors
    all_errors = []
    per_frame_errors = {}  # timestamp -> list of errors
    per_frame_stats = []   # list of dicts with stats per frame

    # Counters for skipped features
    out_of_bounds_count = 0
    invalid_gt_depth_count = 0

    # Get unique timestamps from estimated data
    estimated_timestamps = estimated_df['timestamp'].unique()

    print(f"GT depth frames: {depth_data.len()}")
    print(f"Estimated timestamps with features: {len(estimated_timestamps)}")
    print(f"Total estimated features (left cam): {len(estimated_df)}")

    # Image dimensions
    img_height = depth_data.height
    img_width = depth_data.width

    # Process each estimated timestamp
    for est_ts in estimated_timestamps:
        # Find closest GT frame
        gt_idx = np.argmin(np.abs(gt_timestamps - est_ts))
        gt_ts = gt_timestamps[gt_idx]

        # Check if timestamps are close enough (within 50ms)
        if abs(gt_ts - est_ts) > 0.05:
            print(f"Warning: No close GT match for timestamp {est_ts:.6f}")
            continue

        # Get GT depth image
        gt_depth_img = depth_data.images[gt_idx]

        # Get all features at this timestamp
        features_at_ts = estimated_df[estimated_df['timestamp'] == est_ts]

        frame_errors = []

        for _, row in features_at_ts.iterrows():
            u = row['u']  # column (x-coordinate in image)
            v = row['v']  # row (y-coordinate in image)
            est_depth = row['depth']

            # Round to nearest pixel
            col = int(round(u))
            row_idx = int(round(v))

            # Check bounds
            if row_idx < 0 or row_idx >= img_height or col < 0 or col >= img_width:
                out_of_bounds_count += 1
                continue

            # Get GT depth at (row, col) - NumPy uses row-major order: [row, col]
            gt_depth = gt_depth_img[row_idx, col]

            # Check for invalid GT depth
            if gt_depth <= 0 or np.isinf(gt_depth) or np.isnan(gt_depth) or gt_depth > 200:  # Looking to the infinite horizon
                invalid_gt_depth_count += 1
                continue

            # Skip invalid estimated depths
            if est_depth <= 0 or np.isinf(est_depth) or np.isnan(est_depth):
                raise ValueError(f"Invalid estimated depth ({est_depth}) for feature_id={row['feature_id']} "
                                 f"at timestamp={est_ts}")

            # Compute error (absolute difference)
            error = abs(gt_depth - est_depth)
            frame_errors.append(error)
            all_errors.append(error)

        if frame_errors:
            per_frame_errors[est_ts] = frame_errors
            per_frame_stats.append({
                'timestamp': est_ts,
                'mean': np.mean(frame_errors),
                'median': np.median(frame_errors),
                'count': len(frame_errors)
            })

    # Convert to numpy array for overall stats
    all_errors = np.array(all_errors)

    if len(all_errors) == 0:
        print("No valid depth comparisons found!")
        return

    # Compute overall statistics
    overall_mean = np.mean(all_errors)
    overall_median = np.median(all_errors)
    # Round errors to 2 decimals for mode calculation
    overall_mode_result = stats.mode(np.round(all_errors, 2), keepdims=True)
    overall_mode = overall_mode_result.mode[0]
    overall_std = np.std(all_errors)

    print("\n" + "=" * 60)
    print("OVERALL DEPTH ERROR STATISTICS")
    print("=" * 60)
    print(f"Total valid comparisons: {len(all_errors)}")
    print(f"Out of bounds features skipped: {out_of_bounds_count}")
    print(f"Invalid GT depth features skipped: {invalid_gt_depth_count}")
    print(f"Mean error:   {overall_mean:.4f} meters")
    print(f"Median error: {overall_median:.4f} meters")
    print(f"Mode error:   {overall_mode:.4f} meters")
    print(f"Std dev:      {overall_std:.4f} meters")
    print(f"Min error:    {np.min(all_errors):.4f} meters")
    print(f"Max error:    {np.max(all_errors):.4f} meters")

    # Convert per-frame stats to DataFrame for plotting
    per_frame_df = pd.DataFrame(per_frame_stats)

    # Compute per-frame mode (rounded to 2 decimals)
    per_frame_modes = []
    for ts in per_frame_df['timestamp']:
        frame_errs = per_frame_errors[ts]
        mode_result = stats.mode(np.round(frame_errs, 2), keepdims=True)
        per_frame_modes.append(mode_result.mode[0])
    per_frame_df['mode'] = per_frame_modes

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Per-frame mean error over time
    ax1 = axes[0, 0]
    ax1.plot(per_frame_df['timestamp'], per_frame_df['mean'], 'b-', linewidth=0.8, alpha=0.7)
    ax1.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean: {overall_mean:.4f}m')
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('Mean Depth Error (m)')
    ax1.set_title('Per-Frame Mean Depth Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Per-frame median error over time
    ax2 = axes[0, 1]
    ax2.plot(per_frame_df['timestamp'], per_frame_df['median'], 'g-', linewidth=0.8, alpha=0.7)
    ax2.axhline(y=overall_median, color='r', linestyle='--', label=f'Overall Median: {overall_median:.4f}m')
    ax2.set_xlabel('Timestamp (s)')
    ax2.set_ylabel('Median Depth Error (m)')
    ax2.set_title('Per-Frame Median Depth Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-frame mode error over time
    ax3 = axes[1, 0]
    ax3.plot(per_frame_df['timestamp'], per_frame_df['mode'], 'm-', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=overall_mode, color='r', linestyle='--', label=f'Overall Mode: {overall_mode:.4f}m')
    ax3.set_xlabel('Timestamp (s)')
    ax3.set_ylabel('Mode Depth Error (m)')
    ax3.set_title('Per-Frame Mode Depth Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histogram of all errors with mean, median, mode
    ax4 = axes[1, 1]
    # Cap errors for histogram display (e.g., at 95th percentile)
    error_cap = np.percentile(all_errors, 95)
    capped_errors = np.clip(all_errors, 0, error_cap)
    ax4.hist(capped_errors, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=overall_mean, color='r', linestyle='-', linewidth=2, label=f'Mean: {overall_mean:.4f}m')
    ax4.axvline(x=overall_median, color='g', linestyle='-', linewidth=2, label=f'Median: {overall_median:.4f}m')
    ax4.axvline(x=overall_mode, color='m', linestyle='-', linewidth=2, label=f'Mode: {overall_mode:.4f}m')
    ax4.set_xlabel('Depth Error (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution of Depth Errors (capped at 95th Percentile ({error_cap:.2f}m))')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Depth Error Analysis for ' + robot_name + ' - ' + dataset_num, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_name = 'depth_error_analysis_' + robot_name + '_' + dataset_num + '.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print("\nPlot saved to: " + save_name)
    plt.show()

    # Additional: Error percentiles
    print("\n" + "=" * 60)
    print("ERROR PERCENTILES")
    print("=" * 60)
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(all_errors, p):.4f} meters")


if __name__ == "__main__":
    main()
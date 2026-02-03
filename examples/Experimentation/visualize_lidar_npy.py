#!/usr/bin/env python3
"""
Visualize a LiDAR point cloud from a .npy file.

Usage:
    python visualize_lidar_npy.py <path_to_npy_file>

Example:
    python visualize_lidar_npy.py /path/to/lidar/1234567890.123.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_point_cloud(npy_path: str):
    """
    Visualize a point cloud from a .npy file.

    Args:
        npy_path: Path to the .npy file containing the point cloud
    """

    # Load the point cloud
    points = np.load(npy_path)

    print(f"Loaded point cloud from: {npy_path}")
    print(f"Shape: {points.shape}")
    print(f"Number of points: {points.shape[0]}")
    print(f"Data range:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # Extract timestamp from filename
    filename = Path(npy_path).stem

    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Point Cloud Visualization - Timestamp: {filename}', fontsize=16)

    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View (colored by Z height)')
    plt.colorbar(scatter, ax=ax1, label='Z (m)', shrink=0.5)

    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(points[:, 0], points[:, 1],
                          c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (X-Y, colored by Z)')
    ax2.axis('equal')
    plt.colorbar(scatter2, ax=ax2, label='Z (m)')

    # Side view (X-Z plane)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(points[:, 0], points[:, 2], s=1, alpha=0.6, c='blue')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z)')
    ax3.axis('equal')

    # Front view (Y-Z plane)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(points[:, 1], points[:, 2], s=1, alpha=0.6, c='green')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (Y-Z)')
    ax4.axis('equal')

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_lidar_npy.py <path_to_npy_file>")
        print("\nExample:")
        print("  python visualize_lidar_npy.py /path/to/lidar/1234567890.123.npy")
        sys.exit(1)

    npy_path = sys.argv[1]

    if not Path(npy_path).exists():
        print(f"Error: File not found: {npy_path}")
        sys.exit(1)

    visualize_point_cloud(npy_path)

if __name__ == "__main__":
    main()

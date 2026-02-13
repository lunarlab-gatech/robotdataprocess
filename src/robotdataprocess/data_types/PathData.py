from __future__ import annotations

import colorsys
from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
import copy
from .Data import CoordinateFrame
from .SequentialData import SequentialData
from decimal import Decimal
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
import math
from ..math_utils import interpolate_poses
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from pathlib import Path
from ..ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
from typing import Union, Tuple, List
import tqdm

@typechecked
class PathData(SequentialData):

    positions: np.ndarray # meters (x, y, z)
    orientations: np.ndarray # quaternions (x, y, z, w)
    frame: CoordinateFrame

    def __init__(self, frame_id: str, timestamps: Union[np.ndarray, List], 
                 positions: Union[np.ndarray, List], orientations: Union[np.ndarray, List], 
                 frame: CoordinateFrame):
        super().__init__(frame_id, timestamps)
        self.positions = col_to_dec_arr(positions)
        self.orientations = col_to_dec_arr(orientations)
        self.frame = frame

    def _invalidate_cache(self):
        """ Hook for subclasses to clear cached data after mutations. No-op in PathData. """
        pass

    # =========================================================================
    # ========================= Manipulation Methods ==========================
    # =========================================================================

    def crop_data(self, start: Decimal, end: Union[Decimal, None] = None):
        """ Will crop the data so only values within [start, end] inclusive are kept. """

        # Create boolean mask of data to keep
        mask = ((self.timestamps >= start) & (self.timestamps <= end)) if end is not None else (self.timestamps >= start)

        # Apply mask
        self.timestamps = self.timestamps[mask]
        self.positions = self.positions[mask]
        self.orientations = self.orientations[mask]

        self._invalidate_cache()

    def shift_position(self, x_shift: float, y_shift: float, z_shift: float):
        """
        Shifts the positions of the path.

        Args:
            x_shift (float): Shift in x-axis.
            y_shift (float): Shift in y_axis.
            z_shift (float): Shift in z_axis.
        """
        self.positions[:,0] += Decimal(x_shift)
        self.positions[:,1] += Decimal(y_shift)
        self.positions[:,2] += Decimal(z_shift)

        self._invalidate_cache()

    def shift_to_start_at_identity(self):
        """
        Alter the positions and orientations based so that the first pose
        starts at Identity.
        """

        # Get pose of first robot position w.r.t world
        R_o = R.from_quat(self.orientations[0]).as_matrix()
        T_o = np.expand_dims(self.positions[0], axis=1)

        # Calculate the inverse (pose of world w.r.t first robot location)
        R_inv = R_o.T

        # Rotate positions and orientations
        self.positions = (R_inv @ (self.positions.T - T_o).astype(float)).T
        for i in range(self.len()):
            self.orientations[i] = R.from_matrix((R_inv @ R.from_quat(self.orientations[i]).as_matrix())).as_quat()

        # Convert back to decimal array
        self.positions = col_to_dec_arr(self.positions)
        self.orientations = col_to_dec_arr(self.orientations)

        self._invalidate_cache()

    def interpolate_to_hz(self, target_hz: float):
        """
        Linearly interpolates position and SLERPs orientation so that path data
        is output at the desired frequency.

        Args:
            target_hz (float): Desired output frequency in Hertz (e.g. 6.0)
        """

        # Check that target_hz is valid
        if target_hz <= 0: raise ValueError("target_hz must be positive")

        # Convert timestamps to float seconds
        ts = dec_arr_to_float_arr(self.timestamps)
        pos = dec_arr_to_float_arr(self.positions)
        quat = dec_arr_to_float_arr(self.orientations)

        # Create new evenly spaced timestamps
        duration = ts[-1] - ts[0]
        num_samples = int(np.ceil(duration * target_hz)) + 1
        new_ts = np.linspace(ts[0], ts[-1], num_samples)

        # Interpolate poses
        new_pos, new_quat = interpolate_poses(ts, pos, quat, new_ts)

        # Convert back to Decimal arrays
        self.timestamps = col_to_dec_arr(new_ts)
        self.positions = col_to_dec_arr(new_pos)
        self.orientations = col_to_dec_arr(new_quat)

        self._invalidate_cache()

    # =========================================================================
    # =========================== Frame Conversions ===========================
    # =========================================================================

    def to_FLU_frame(self):
        # If we are already in the FLU frame, return
        if self.frame == CoordinateFrame.FLU:
            print("Data already in FLU coordinate frame, returning...")
            return

        # If in NED, run the conversion
        elif self.frame == CoordinateFrame.NED:
            # Define the rotation matrix
            R_NED = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]])

            # Do a change of basis to update the frame
            self._convert_frame(R_NED)

            # Update frame
            self.frame = CoordinateFrame.FLU

            self._invalidate_cache()

        # Otherwise, throw an error
        else:
            raise RuntimeError(f"PathData class is in an unexpected frame: {self.frame}!")

    def apply_transformation_left_side(self, H: np.ndarray):
        """
        Applies a rigid-body transformation to the entire path.
        In terms of transformation matrices, this multiplies this path
        on the left side.

        Args:
            H: The 4x4 transformation matrix
        """

        # Apply the transformation
        H_self = np.eye(4).reshape(1, 4, 4).repeat(self.len(), axis=0)  # shape (N,4,4)
        H_self[:, :3, :3] = R.from_quat(self.orientations).as_matrix()
        H_self[:, :3, 3] = self.positions
        H_output = H @ H_self

        # Extract results and save
        self.positions = H_output [:, :3, 3]
        self.orientations = R.from_matrix(H_output[:, :3, :3]).as_quat()

        self._invalidate_cache()

    def apply_transformation_right_side(self, H: np.ndarray):
        """
        Applies a rigid-body transformation to the entire path.
        In terms of transformation matrices, this multiplies this path
        on the right side (row-vector convention).

        Args:
            H: The 4x4 transformation matrix
        """

        # Apply the transformation
        H_self = np.eye(4).reshape(1, 4, 4).repeat(self.len(), axis=0)  # shape (N,4,4)
        H_self[:, :3, :3] = R.from_quat(self.orientations).as_matrix()
        H_self[:, :3, 3] = self.positions
        H_output = H_self @ H

        # Extract results and save
        self.positions = H_output [:, :3, 3]
        self.orientations = R.from_matrix(H_output[:, :3, :3]).as_quat()

        self._invalidate_cache()

    def _convert_frame(self, R_frame: np.ndarray):
        """ Uses a change of basis to update the positions and orientations. """
        R_frame_Q = R.from_matrix(R_frame)
        self.positions = (R_frame @ self.positions.T).T
        self._ori_change_of_basis(R_frame_Q)

        self._invalidate_cache()

    def _ori_apply_rotation(self, R_i: R):
        """ Applies a rotation (not a change of basis) to orientations, thus stays in the same frame. """
        for i in range(self.len()):
            self.orientations[i] = (R_i * R.from_quat(self.orientations[i])).as_quat()

        self._invalidate_cache()

    def _ori_change_of_basis(self, R_i: R):
        """ Applies a change of basis to orientations """
        for i in range(self.len()):
            self.orientations[i] = (R_i * R.from_quat(self.orientations[i]) * R_i.inv()).as_quat()

        self._invalidate_cache()

    # =========================================================================
    # ============================ Class Methods ==============================
    # =========================================================================

    @classmethod
    def from_ros2_bag(cls, bag_path: Union[Path, str], odom_topic: str, frame: CoordinateFrame) -> PathData:
        """
        Creates a class structure from a ROS2 bag file with a Path topic.

        Args:
            bag_path (Union[Path, str]): Path to the ROS2 bag file.
            odom_topic (str): Topic of the Path messages.
            frame: The coordinate frame of this data.
        Returns:
            PathData: Instance of this class.
        """

        # Get topic message count and typestore
        bag_wrapper = Ros2BagWrapper(bag_path, None)
        typestore: Typestore = bag_wrapper.get_typestore()
        num_msgs: int = bag_wrapper.get_topic_count(odom_topic)
        
        # Make empty arrays
        timestamps_np = np.zeros(0, dtype=Decimal)
        positions_np = np.zeros((0, 3), dtype=Decimal)
        orientations_np = np.zeros((0, 4), dtype=Decimal)

        # Setup tqdm bar & counter
        pbar = tqdm.tqdm(total=num_msgs, desc="Extracting Path...", unit=" msgs")

        # Extract the odometry information
        frame_id = None
        with Reader2(str(bag_path)) as reader:

            # Extract frame_id from first message
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):  
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                frame_id = msg.header.frame_id
                break

            # Extract message data
            connections = [x for x in reader.connections if x.topic == odom_topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                
                # NOTE: Currently, this method doesn't track when each Path message 
                # is recieved, and throws away duplicate poses contained in multiple
                # Path messages.

                # Iterate through each pose in the message
                for pose in msg.poses:
                    
                    # See if we already have this pose (via timestamp)
                    ts = bag_wrapper.extract_timestamp(pose)
                    if ts in timestamps_np:
                        continue

                    # If not, extract data
                    timestamps_np = np.concatenate((timestamps_np, [ts]), axis= 0)
                    pos = pose.pose.position
                    positions_np = np.concatenate((positions_np, [[Decimal(pos.x), Decimal(pos.y), Decimal(pos.z)]]), axis=0)
                    ori = pose.pose.orientation
                    orientations_np = np.concatenate((orientations_np, [[Decimal(ori.x), Decimal(ori.y), Decimal(ori.z), Decimal(ori.w)]]), axis=0)

                    # Increment the count
                    pbar.update(1)

        # Create an OdometryData class
        return cls(frame_id, timestamps_np, positions_np, orientations_np, frame)
    
    @classmethod
    def from_evo(cls, pose_trajectory_3d: PoseTrajectory3D, frame_id: str, frame: CoordinateFrame) -> PathData:
        """ Creates a PathData object from an evo PoseTrajectory3D object. """

        # Convert orientations from wxyz to xyzw
        orientations_xyzw = pose_trajectory_3d.orientations_quat_wxyz[:, [1, 2, 3, 0]]

        return cls(frame_id=frame_id, 
                   timestamps=pose_trajectory_3d.timestamps, 
                   positions=pose_trajectory_3d.positions_xyz, 
                   orientations=orientations_xyzw,
                   frame=frame)
    
    # =========================================================================
    # ============================ Visualization ============================== 
    # =========================================================================  

    @staticmethod
    def visualize_2D(dataList: List[PathData], isGTList: List[bool], colorList: List[str], nameList: List[str], 
                     save_path: Union[str, None] = None, no_background: bool = False, line_width: float = 1.0, 
                     show_grid: bool = False, legend: bool = True,
                     no_border: bool = False, disable_x_label: bool = False, disable_y_label: bool = False,
                     google_maps_scale_bar: bool = False, google_maps_scale_bar_loc: str ="bottom-right",
                     gt_color_lightness_range_val: int = 3,
                     background_image_path: str | None = None,
                     background_image_x_edge: float | None = None, ax: plt.Axes | None = None,):
        """
        Plot all PathData objects on a 2D XY plane.
        
        Args:
            dataList: All PathData objects to plot.
            isGTList: Whether or not each PathData object is GT.
            colorList: Colors to assign to each PathData object (as hex strings starting with #).
            nameList: Robot names corresponding to each PathData object.
            save_path: If provided, figure will be saved to the location. Otherwise, show the figure.
                This does nothing if ax is not None.
            no_background: If true, plot with a transparent background.
            line_width: Width of trajectory lines in the plot.
            show_grid: Whether to draw a grid on the plot.
            legend: If true, include a legend.
            no_border: If true, remove the x & y axes.
            disable_x_label: Remove the x label.
            disable_y_label: Remove the y label.
            google_maps_scale_bar: Whether to add a Google Maps-like scale bar onto the plot.
            google_maps_scale_bar_loc: Location for the scale bar.
            background_image_path: Path to an image to plot in the background; It is assumed
                that the center of the image corresponds to x=0 & y=0 in the PataData frames.
            background_image_x_edge: The distance in meters from center of image to the x edge.
            ax: If passed, plot is drawn onto these axes instead of on a new figure.
        """

        # Check lengths of arguments
        if len(dataList) != len(isGTList) or len(dataList) != len(colorList) or len(dataList) != len(nameList):
            raise ValueError("Lengths of all Lists must be equal!")
        num_data_objs = len(dataList)

        # Check other argument requirements
        if gt_color_lightness_range_val < 0 or gt_color_lightness_range_val >= 20:
            raise ValueError("gt_color_lightness_range_val must be between 0 and 19 inclusive!")

        # Convert hex colors to a palette with varying lightness
        paletteList = []
        for c in colorList:
            # Convert base color to HLS
            rgb = mcolors.to_rgb(c)
            h, _, s = colorsys.rgb_to_hls(*rgb)

            # Generate similar colors with varying lightness
            lightnesses = np.linspace(0.0, 1.0, 20)
            paletteList.append([colorsys.hls_to_rgb(h, li, s) for li in lightnesses])

        # Create the figure or use the provided axes
        created_fig = False
        if ax is None:
            fig, axs = plt.subplots(1, 1)
            created_fig = True
        else:
            axs = ax
            fig = axs.figure
        if no_background:
            fig.patch.set_facecolor('white')
            axs.set_facecolor('none')
        else:
            fig.patch.set_facecolor('white')
            axs.set_facecolor("#F0F0F0")

        # Draw background image
        if background_image_path is not None:
            img = mpimg.imread(background_image_path)
            if background_image_x_edge:
                x_extent_meters = background_image_x_edge
                h, w = img.shape[0], img.shape[1]
                y_extent_meters = x_extent_meters / w * h
                extent = [-x_extent_meters, x_extent_meters, -y_extent_meters, y_extent_meters]
                axs.imshow(img, extent=extent, origin="upper", alpha=1.0, zorder=0)
            else:
                raise ValueError("Extent must be provided with Background image size via background_image_x_edge.")

        # Calculate trajectory bounds
        all_x = np.concatenate([dec_arr_to_float_arr(path.positions[:,0]) for path in dataList])
        all_y = np.concatenate([dec_arr_to_float_arr(path.positions[:,1]) for path in dataList])
        padding_x = (all_x.max() - all_x.min()) * 0.05
        padding_y = (all_y.max() - all_y.min()) * 0.05
        x_min, x_max = all_x.min() - padding_x, all_x.max() + padding_x
        y_min, y_max = all_y.min() - padding_y, all_y.max() + padding_y

        # Plot the trajectories
        for i in range(num_data_objs):
            label = nameList[i] + (" (GT)" if isGTList[i] else " (Est.)")
            linestyle = ("dotted" if isGTList[i] else None)
            color = (paletteList[i][gt_color_lightness_range_val] if isGTList[i] else paletteList[i][9])
            axs.plot(dataList[i].positions[:,0], dataList[i].positions[:,1], 
                     label=label, color=color, linewidth=line_width, linestyle=linestyle)
    
        # Calculate the current aspect ratio and make it match the target
        target_ar = 1.5  
        current_width = x_max - x_min
        current_height = y_max - y_min
        current_ar = current_width / current_height

        if current_ar > target_ar:
            target_height = current_width / target_ar
            diff = target_height - current_height
            y_min -= diff / 2
            y_max += diff / 2
        elif current_ar < target_ar:
            target_width = current_height * target_ar
            diff = target_width - current_width
            x_min -= diff / 2
            x_max += diff / 2

        # Set the new limits that respect the aspect ratio
        axs.set_xlim(x_min, x_max)
        axs.set_ylim(y_min, y_max)
        axs.set_aspect('equal', adjustable='box')
        
        # Make the tick spacing match
        yticks = axs.get_yticks()
        if len(yticks) > 1:
            y_spacing = yticks[1] - yticks[0]
            axs.xaxis.set_major_locator(MultipleLocator(y_spacing))

        # Adjust the spines
        for spine in axs.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.75)

        # Add Grid if Desired
        if show_grid:
            axs.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        else:
            axs.grid(False)

        # Add labels
        if not disable_x_label:
            axs.set_xlabel("X (meters)")
        if not disable_y_label:
            axs.set_ylabel("Y (meters)")

        # Additional configurable parameters
        if legend:
            axs.legend()
        if no_border:
            axs.set_axis_off()

        # Helper function to make a Google Maps like Scale Bar
        def add_google_maps_scale(ax, length_m: float, location: str):
            # Metric calculations
            metric_label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m/1000:.1g} km"
            
            # American calculations
            exact_ft = length_m * 3.28084
            if exact_ft < 5280:
                rounded_ft = int(exact_ft // 25) * 25
                if rounded_ft == 0: 
                    rounded_ft = 25 
                american_label = f"{rounded_ft} ft"
                american_width_m = rounded_ft / 3.28084
            else:
                miles = exact_ft / 5280
                rounded_mi = math.floor(miles * 10) / 10.0
                if rounded_mi == 0: rounded_mi = 0.1
                american_label = f"{rounded_mi:.1f} mi"
                american_width_m = rounded_mi * 1609.34

            # 3. Geometry Setup
            xmin, xmax = ax.get_xlim()
            range_m = xmax - xmin
            
            tick_h = 0.025
            if location == "bottom-right":
                end_x = 0.93
                y_center = 0.12
            elif location == "top-right":
                end_x = 0.93
                y_center = 0.88
            else:
                raise ValueError("location must be 'top-right' or 'bottom-right'")
            
            # Fractional widths for both bars
            frac_m = length_m / range_m
            frac_am = american_width_m / range_m

            # 4. Styling
            main_color = "#373737"
            pe_text = [path_effects.withStroke(linewidth=2.5, foreground='white')]
            pe_line = [path_effects.withStroke(linewidth=5, foreground='white')]
            
            line_params = {'color': main_color, 'transform': ax.transAxes, 'lw': 3.0, 
                        'path_effects': pe_line, 'solid_capstyle': 'round', 'zorder': 10}
            text_params = {'color': main_color, 'transform': ax.transAxes, 'ha': 'right', 
                        'path_effects': pe_text, 'fontsize': 18, 'weight': 'black', 'zorder': 11}

            # --- DRAW METRIC (TOP) ---
            m_start_x = end_x - frac_m
            am_start_x = end_x - frac_am
            # Horizontal and L-brackets (upward)
            ax.plot([m_start_x, end_x], [y_center, y_center], **line_params)
            ax.plot([m_start_x, m_start_x], [y_center, y_center + tick_h], **line_params)
            ax.plot([end_x, end_x], [y_center, y_center + tick_h], **line_params)
            ax.text(am_start_x + 0.9 * (end_x - am_start_x), y_center + 0.01, metric_label, va='bottom', **text_params)

            # --- DRAW AMERICAN (BOTTOM) ---
            # Horizontal and L-brackets (downward)
            ax.plot([am_start_x, end_x], [y_center, y_center], **line_params)
            ax.plot([am_start_x, am_start_x], [y_center, y_center - tick_h], **line_params)
            ax.plot([end_x, end_x], [y_center, y_center - tick_h], **line_params)
            ax.text(am_start_x + 0.9 * (end_x - am_start_x), y_center - 0.018, american_label, va='top', **text_params)

            # Re-draw grey lines on top to clean up the white stroke overlaps
            clean_params = {'color': main_color, 'transform': ax.transAxes, 'lw': 2.5, 'zorder': 12}
            ax.plot([m_start_x, end_x], [y_center, y_center], **clean_params)
            ax.plot([m_start_x, m_start_x], [y_center, y_center + tick_h], **clean_params)
            ax.plot([end_x, end_x], [y_center, y_center + tick_h], **clean_params)
            ax.plot([am_start_x, end_x], [y_center, y_center], **clean_params)
            ax.plot([am_start_x, am_start_x], [y_center, y_center - tick_h], **clean_params)
            ax.plot([end_x, end_x], [y_center, y_center - tick_h], **clean_params)

        # Draw a google maps scale bar
        if google_maps_scale_bar:
            range_m = x_max - x_min
            target_length = range_m * 0.25
            if target_length < 10: suggested_length = target_width
            else: suggested_length = int(round(target_length / 10.0)) * 10
            add_google_maps_scale(axs, suggested_length, google_maps_scale_bar_loc)

        # Save/Plot the results
        if created_fig:
            if save_path is not None:
                fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0)
            else:
                plt.show()
            plt.close(fig)

        return axs

    def visualize_3D(self, otherList: List[PathData], titles: List[str], axes_length: Union[float, List[float]] = 10.0, axes_interval: Union[int, List[int]] = 1000):
        """
        Visualizes this PathData (and all others included in otherList) on a single plot.

        Args:
            otherList (List[PathData]): All other PathData objects whose path should also be visualized on this plot.
            titles (List[str]): Titles for each PathData object, starting with self.
        """

        def draw_axes(data: PathData, axes_length: int, axes_interval: int):
            """Helper function that visualizes orientation along the trajectory path with axes."""

            for i in range(0, data.len(), axes_interval):
                # Extract data
                pos = data.positions[i].astype(np.float64)
                quat = data.orientations[i].astype(np.float64)
                rot = R.from_quat(quat)

                # Define unit vectors for X, Y, Z in local frame
                x_axis = rot.apply([1, 0, 0])
                y_axis = rot.apply([0, 1, 0])
                z_axis = rot.apply([0, 0, 1])

                # Plot axes
                ax.quiver(*pos, *x_axis, length=axes_length, color='r', normalize=True, linewidth=0.8)
                ax.quiver(*pos, *y_axis, length=axes_length, color='g', normalize=True, linewidth=0.8)
                ax.quiver(*pos, *z_axis, length=axes_length, color='b', normalize=True, linewidth=0.8)

        # Ensure that the lists are of the proper sizes
        if (len(otherList) + 1) != len(titles):
            raise ValueError("Length of titles should be one more than length of otherlist!")

        # Build a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.positions[:,0].astype(np.float64), 
                self.positions[:,1].astype(np.float64), 
                self.positions[:,2].astype(np.float64), label=titles[0])
        for i, other in enumerate(otherList):
            ax.plot(other.positions[:,0].astype(np.float64), 
                    other.positions[:,1].astype(np.float64), 
                    other.positions[:,2].astype(np.float64), 
                    label=titles[1+i])
            
        # Handle axes_length and axes_interval if they are lists
        if isinstance(axes_length, list):
            if len(axes_length) != (len(otherList) + 1):
                raise ValueError("If axes_length is a list, it must be the same length as otherList + 1!")
        else: axes_length: list[float] = [axes_length] * (len(otherList) + 1)

        if isinstance(axes_interval, list):
            if len(axes_interval) != (len(otherList) + 1):
                raise ValueError("If axes_interval is a list, it must be the same length as otherList + 1!")
        else: axes_interval: list[int] = [axes_interval] * (len(otherList) + 1)

        # Draw orientation axes (X = red, Y = green, Z = blue)
        draw_axes(self, axes_length=axes_length[0], axes_interval=axes_interval[0])
        for i, other in enumerate(otherList):
            draw_axes(other, axes_length=axes_length[i+1], axes_interval=axes_interval[i+1])

        # Set labels
        ax.set_title("Trajectory Comparison with Full Orientation")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        # Concatenate all x, y and z values together
        all_x = self.positions[:,0]
        all_y = self.positions[:,1]
        all_z = self.positions[:,2]
        for other in otherList:
            all_x = np.concatenate((all_x, other.positions[:,0]))
            all_y = np.concatenate((all_y, other.positions[:,1]))
            all_z = np.concatenate((all_z, other.positions[:,2]))
        all_x = all_x.astype(np.float64)
        all_y = all_y.astype(np.float64)
        all_z = all_z.astype(np.float64)

        # Set an equal scale for all axes
        x_center = (all_x.max() + all_x.min()) / 2
        y_center = (all_y.max() + all_y.min()) / 2
        z_center = (all_z.max() + all_z.min()) / 2
        max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()) / 2
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)

        # Show the plot
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # ============================ Export Methods ============================= 
    # =========================================================================  

    def to_OdometryData(self, new_frame_id: str, new_child_frame_id: str):
        """ 
        Returns an OdometryData object for this class. 

        Parameters:
            new_frame_id: The new frame ID to assign to the OdometryData object. 
            new_child_frame_id: The new child frame ID to assign to the OdometryData object.
        """

        from .OdometryData import OdometryData
        return OdometryData(frame_id=new_frame_id,
                            child_frame_id=new_child_frame_id,
                            timestamps=self.timestamps,
                            positions=self.positions,
                            orientations=self.orientations,
                            frame=self.frame)

    def to_evo(self) -> PoseTrajectory3D:
        """ Returns an evo PoseTrajectory3D object for this class. """

        orientations_wxyz = dec_arr_to_float_arr(self.orientations[:, [3, 0, 1, 2]])
        return PoseTrajectory3D(positions_xyz=dec_arr_to_float_arr(self.positions), 
                                orientations_quat_wxyz=orientations_wxyz,
                                timestamps=dec_arr_to_float_arr(self.timestamps))
    
    # =========================================================================
    # ======================= Multi PathData Methods ========================== 
    # ========================================================================= 

    @staticmethod
    def make_start_and_end_times_match(est: list[PathData], gt: list[PathData]) -> Tuple[list[PathData], list[PathData]]:
        """ 
        For pairs of lists of PathData objects, extract each pair by index and 
        ensure that the first and last timestamps match by extending the data
        as necessary at the start and end with duplicate values. Used for evaluation
        purposes.
        
        Mimics the behavior found in ROMAN's (https://github.com/lunarlab-gatech/roman) evaluation scripts.

        Parameters:
            est: List of PathData objects that represent estimated paths.
            gt: List of PathData objects that represent ground truth paths.
        """

        # Copy the PathData objects so we don't modify the originals
        est = copy.deepcopy(est)
        gt = copy.deepcopy(gt)

        # Check that the lists are the same length
        if len(est) == 0 or len(gt) == 0 or len(est) != len(gt):
            raise ValueError("est and gt lists must be non-empty and of the same length!")

        # For each pair of PathData objects
        for est_i, gt_i in zip(est, gt):

            # Adjust start times
            if est_i.timestamps[0] < gt_i.timestamps[0]:
                gt_i.timestamps = np.concatenate(([est_i.timestamps[0]], gt_i.timestamps))
                gt_i.positions = np.concatenate(([gt_i.positions[0]], gt_i.positions))
                gt_i.orientations = np.concatenate(([gt_i.orientations[0]], gt_i.orientations))
            elif est_i.timestamps[0] > gt_i.timestamps[0]:
                est_i.timestamps = np.concatenate(([gt_i.timestamps[0]], est_i.timestamps))
                est_i.positions = np.concatenate(([est_i.positions[0]], est_i.positions))
                est_i.orientations = np.concatenate(([est_i.orientations[0]], est_i.orientations))

            # Adjust end times
            if est_i.timestamps[-1] < gt_i.timestamps[-1]:
                est_i.timestamps = np.concatenate((est_i.timestamps, [gt_i.timestamps[-1]]))
                est_i.positions = np.concatenate((est_i.positions, [est_i.positions[-1]]))
                est_i.orientations = np.concatenate((est_i.orientations, [est_i.orientations[-1]]))
            elif est_i.timestamps[-1] > gt_i.timestamps[-1]:
                gt_i.timestamps = np.concatenate((gt_i.timestamps, [est_i.timestamps[-1]]))
                gt_i.positions = np.concatenate((gt_i.positions, [gt_i.positions[-1]]))
                gt_i.orientations = np.concatenate((gt_i.orientations, [gt_i.orientations[-1]]))
        
        # Return the modified lists
        return est, gt

    @staticmethod
    def concatenate_PathData(path_data_objs: list[PathData]) -> PathData:
        """ 
        Combines multiple PathData objects into a single PathData object. In doing so,
        will shift the timestamps of each subsequent PathData so that their data starts
        one second after the previous PathData ends. Also assumes the frame_id and frame
        of the first PathData object for final PathData object.
        
        Mimics the behavior found in ROMAN's (https://github.com/lunarlab-gatech/roman) evaluation scripts.
        """

        print("Warning! This code has not been unit tested yet!")

        if len(path_data_objs) == 0:
            raise ValueError("path_data_objs list is empty!")
        if len(path_data_objs) == 1:
            raise ValueError("path_data_objs list has only one element; no need to concatenate!")

        # NOTE: Assumes the frame_id and frame of the first object
        frame_id = path_data_objs[0].frame_id
        frame = path_data_objs[0].frame

        # Create all empty arrays to hold concatenated data
        all_timestamps = np.zeros((0,), dtype=Decimal)
        all_positions = np.zeros((0, 3), dtype=Decimal)
        all_orientations = np.zeros((0, 4), dtype=Decimal)

        # For each PathData object
        for i, path_data in enumerate(path_data_objs):

            # If not first PathData, shift timestamps
            if i == 0:
                shifted_timestamps = path_data.timestamps
            else:
                shifted_timestamps = path_data.timestamps - path_data.timestamps[0] + all_timestamps[-1] + 1

            # Concatentate data
            all_timestamps = np.concatenate((all_timestamps, shifted_timestamps), axis=0)
            all_positions = np.concatenate((all_positions, path_data.positions), axis=0)
            all_orientations = np.concatenate((all_orientations, path_data.orientations), axis=0)

        return PathData(frame_id, all_timestamps, all_positions, all_orientations, frame)
    
    @staticmethod
    def seperate_PathData(original_PathDatas: list[PathData], merged_PathData: PathData) -> list[PathData]:
        """
        Inverse of concatenate_PathData(); needs the original objects to know the timestamps
        that each trajectory is found on.
        """

        # Calculate the start and end times for the beginning of each robots data
        start_times: list[float] = []
        end_times: list[float] = []
        for i in range(len(original_PathDatas)): 
            pd: PathData = original_PathDatas[i]
            if i == 0:
                start_times.append(pd.timestamps[0])
                end_times.append(pd.timestamps[-1])
            else:
                start_times.append(end_times[-1] + 1)
                end_times.append(pd.timestamps[-1] - pd.timestamps[0] + end_times[-1] + 1)

        # Make deep copies of trajectories
        merged_PathData_copies: list[PathData] = [copy.deepcopy(merged_PathData) for _ in range(len(original_PathDatas))]

        # Reduce trajectories to the specific time that covers each robot
        for i, pd in enumerate(merged_PathData_copies):
            pd.crop_data(Decimal(start_times[i]), Decimal(end_times[i]))
        return merged_PathData_copies

    @staticmethod
    def align_and_calculate_traj_errors(gt_path: PathData, est_path: PathData, max_diff: float, visualize: bool = False, 
            axes_length: Union[float, list[float]] = 10.0, axes_interval: Union[int, list[int]] = 1000) -> Tuple[dict, PathData, PathData]:
        """
        Utilizing the evo library, calculates a variety of trajectory error metrics
        and returns them in a dictionary. Also returns aligned PathData objects.

        Parameters:
            max_diff: maximum absolute time difference allowed between associated timestamps
            visualize: If true, will show a 3D plot of the aligned trajectories.
            axes_length: Same as in visualize() method.
            axes_interval: Same as in visualize() method.
        """

        gt_traj: PoseTrajectory3D = gt_path.to_evo()
        est_traj: PoseTrajectory3D = est_path.to_evo()

        gt_traj, est_traj = sync.associate_trajectories(gt_traj, est_traj, max_diff)


        est_traj_align: PoseTrajectory3D = copy.deepcopy(est_traj)
        est_traj_align.align(gt_traj, correct_scale=False, correct_only_scale=False) 

        path_pair: Tuple[PoseTrajectory3D, PoseTrajectory3D] = (gt_traj, est_traj_align)

        # Calculate various error metrics using evo, including APE and RPE
        all_pose_relations: List[metrics.PoseRelation] = [metrics.PoseRelation.full_transformation, # dimensionless
                                                          metrics.PoseRelation.translation_part, # meters
                                                          metrics.PoseRelation.rotation_part, # dimensionless
                                                          metrics.PoseRelation.rotation_angle_deg, # degrees
                                                          metrics.PoseRelation.rotation_angle_rad, # radians
                                                          metrics.PoseRelation.point_distance, # meters
                                                          metrics.PoseRelation.point_distance_error_ratio] # percent
        all_statistic_types: List[metrics.StatisticsType] = [metrics.StatisticsType.rmse,
                                                             metrics.StatisticsType.mean,
                                                             metrics.StatisticsType.median,
                                                             metrics.StatisticsType.std,
                                                             metrics.StatisticsType.min,
                                                             metrics.StatisticsType.max,
                                                             metrics.StatisticsType.sse]
        all_metrics: List = [metrics.APE, metrics.RPE]
        dict_all_results: dict = {}
        for metric in all_metrics:
            dict_metric: dict = {}

            for pose_relation in all_pose_relations:
                dict_relation: dict = {}

                # Skip uncompatible relation with metric
                if metric is metrics.APE and pose_relation == metrics.PoseRelation.point_distance_error_ratio:
                    continue

                path_pair_copied = copy.deepcopy(path_pair)
                metric_with_relation: metrics.PE = metric(pose_relation)
                metric_with_relation.process_data(path_pair_copied)

                for stat in all_statistic_types:
                    final_stat: float = metric_with_relation.get_statistic(stat)
                    dict_relation[stat.name] = final_stat

                dict_metric[pose_relation.name] = dict_relation
            
            dict_all_results[metric.__name__] = dict_metric
            
        # Convert the aligned trajectory data back to PathData objects
        est_traj_align_pathdata = PathData.from_evo(est_traj_align, est_path.frame_id, est_path.frame)
        gt_traj_pathdata = PathData.from_evo(gt_traj, gt_path.frame_id, gt_path.frame)

        # Visualize the aligned trajectories if desired
        if visualize:
            gt_traj_pathdata.visualize_3D([est_traj_align_pathdata], ['Ground Truth', 'Estimated (Aligned)'], 
                              axes_interval=axes_interval, axes_length=axes_length)
        
        return dict_all_results, est_traj_align_pathdata, gt_traj_pathdata
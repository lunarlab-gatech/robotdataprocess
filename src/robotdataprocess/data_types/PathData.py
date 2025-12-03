from __future__ import annotations

from ..conversion_utils import col_to_dec_arr, dec_arr_to_float_arr
import copy
from .Data import Data, CoordinateFrame
from decimal import Decimal
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..rosbag.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys.store import Typestore
from scipy.spatial.transform import Rotation as R
from typeguard import typechecked
import tqdm

class PathData(Data):

    positions: np.ndarray[Decimal] # meters (x, y, z)
    orientations: np.ndarray[Decimal] # quaternions (x, y, z, w)
    frame: CoordinateFrame

    @typechecked
    def __init__(self, frame_id: str, timestamps: np.ndarray | list, 
                 positions: np.ndarray | list, orientations: np.ndarray | list, 
                 frame: CoordinateFrame):
        super().__init__(frame_id, timestamps)
        self.positions = col_to_dec_arr(positions)
        self.orientations = col_to_dec_arr(orientations)
        self.frame = frame

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    @typechecked
    def from_ros2_bag(cls, bag_path: Path | str, odom_topic: str, frame: CoordinateFrame) -> PathData:
        """
        Creates a class structure from a ROS2 bag file with a Path topic.

        Args:
            bag_path (Path | str): Path to the ROS2 bag file.
            odom_topic (str): Topic of the Path messages.
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
        
        print("Warning! This code has not been unit tested yet!")

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

    @typechecked
    def visualize(self, otherList: list[PathData], titles: list[str], axes_length: float | list[float] = 10.0, axes_interval: int | list[int] = 1000):
        """
        Visualizes this PathData (and all others included in otherList)
        on a single plot.

        Args:
            otherList (list[PathData]): All other PathData objects whose path should also be visualized on this plot.
            titles (list[str]): Titles for each PathData object, starting with self.

        """

        print("Warning! This code has not been unit tested yet!")

        def draw_axes(data: PathData, axes_length: int, axes_interval: int):
            """Helper function that visualizes orientation along the trajectory path with axes."""

            for i in range(0, data.len(), axes_interval):
                # Extract data
                pos = data.positions[i].astype(np.float64)
                quat = data.orientations[i].astype(np.float64)
                rot = R.from_quat(quat, scalar_first=False)

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

    def to_OdometryData(self, new_frame_id: str):
        """ 
        Returns an OdometryData object for this class. 

        Parameters:
            new_frame_id: The new frame ID to assign to the OdometryData object. 
                The previous frame ID of this PathData object will be used as the
                child_frame_id of the OdometryData object.
        """

        print("Warning! This code has not been unit tested yet!")

        from .OdometryData import OdometryData
        return OdometryData(frame_id=new_frame_id,
                            child_frame_id=self.frame_id,
                            timestamps=self.timestamps,
                            positions=self.positions,
                            orientations=self.orientations,
                            frame=self.frame)

    def to_evo(self) -> PoseTrajectory3D:
        """ Returns an evo PoseTrajectory3D object for this class. """

        print("Warning! This code has not been unit tested yet!")

        orientations_wxyz = dec_arr_to_float_arr(self.orientations[:, [3, 0, 1, 2]])
        return PoseTrajectory3D(positions_xyz=dec_arr_to_float_arr(self.positions), 
                                orientations_quat_wxyz=orientations_wxyz,
                                timestamps=dec_arr_to_float_arr(self.timestamps))
    
    # =========================================================================
    # ======================= Multi PathData Methods ========================== 
    # ========================================================================= 

    @staticmethod
    def calculate_trajectory_errors(gt_path: PathData, est_path: PathData, max_diff: float, visualize: bool = False, 
                                    axes_length: float | list[float] = 10.0, axes_interval: int | list[int] = 1000) -> dict:
        """
        Utilizing the evo library, calculates a variety of trajectory error metrics
        and returns them in a dictionary.

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

        path_pair: tuple[PoseTrajectory3D, PoseTrajectory3D] = (gt_traj, est_traj_align)

        # Calculate various error metrics using evo, including APE and RPE
        all_pose_relations: list[metrics.PoseRelation] = [metrics.PoseRelation.full_transformation, # dimensionless
                                                          metrics.PoseRelation.translation_part, # meters
                                                          metrics.PoseRelation.rotation_part, # dimensionless
                                                          metrics.PoseRelation.rotation_angle_deg, # degrees
                                                          metrics.PoseRelation.rotation_angle_rad, # radians
                                                          metrics.PoseRelation.point_distance, # meters
                                                          metrics.PoseRelation.point_distance_error_ratio] # percent
        all_statistic_types: list[metrics.StatisticsType] = [metrics.StatisticsType.rmse,
                                                             metrics.StatisticsType.mean,
                                                             metrics.StatisticsType.median,
                                                             metrics.StatisticsType.std,
                                                             metrics.StatisticsType.min,
                                                             metrics.StatisticsType.max,
                                                             metrics.StatisticsType.sse]
        all_metrics: list[metrics.PE] = [metrics.APE, metrics.RPE]
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

        # Visualize the aligned trajectories if desired
        if visualize:
            
            # Convert est_traj_align back to PathData
            est_traj_align_pathdata: PathData = PathData.from_evo(est_traj_align, gt_path.frame_id, gt_path.frame)

            # Visualize the aligned trajectories with specified axes length and interval
            gt_path.visualize([est_traj_align_pathdata], ['Ground Truth', 'Estimated (Aligned)'], 
                              axes_interval=axes_interval, axes_length=axes_length)

        return dict_all_results
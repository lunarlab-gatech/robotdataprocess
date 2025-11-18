from __future__ import annotations

from ..conversion_utils import convert_collection_into_decimal_array, convert_decimal_array_into_float_array
import copy
from .Data import Data
from decimal import Decimal
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from pathlib import Path
from ..rosbag.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys.store import Typestore
from typeguard import typechecked
import tqdm

class PathData(Data):

    positions: np.ndarray[Decimal] # meters (x, y, z)
    orientations: np.ndarray[Decimal] # quaternions (x, y, z, w)

    @typechecked
    def __init__(self, frame_id: str, timestamps: np.ndarray | list, 
                 positions: np.ndarray | list, orientations: np.ndarray | list):
        super().__init__(frame_id, timestamps)
        self.positions = convert_collection_into_decimal_array(positions)
        self.orientations = convert_collection_into_decimal_array(orientations)

    # =========================================================================
    # ============================ Class Methods ============================== 
    # =========================================================================  

    @classmethod
    @typechecked
    def from_ros2_bag(cls, bag_path: Path | str, odom_topic: str):
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
        return cls(frame_id, timestamps_np, positions_np, orientations_np)
    
    # =========================================================================
    # ============================ Export Methods ============================= 
    # =========================================================================  

    def to_evo(self) -> PoseTrajectory3D:
        """ Returns an evo PoseTrajectory3D object for this class. """

        orientations_wxyz = convert_decimal_array_into_float_array(self.orientations[:, [3, 0, 1, 2]])
        return PoseTrajectory3D(positions_xyz=convert_decimal_array_into_float_array(self.positions), 
                                orientations_quat_wxyz=orientations_wxyz,
                                timestamps=convert_decimal_array_into_float_array(self.timestamps))
    
    # =========================================================================
    # ======================= Multi PathData Methods ========================== 
    # ========================================================================= 

    @staticmethod
    def calculate_trajectory_errors(gt_path: PathData, est_path: PathData, max_diff: float) -> dict:
        """
        Utilizing the evo library, calculates a variety of trajectory error metrics
        and returns them in a dictionary.

        Parameters:
            max_diff: maximum absolute time difference allowed between associated timestamps
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

        return dict_all_results
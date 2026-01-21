from collections import defaultdict
from .data_types.Data import CoordinateFrame
from .data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from .data_types.OdometryData import OdometryData
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .ros.Ros2BagWrapper import Ros2BagWrapper
from rosbags.rosbag2 import Reader as Reader2
from rosbags.rosbag2 import Writer as Writer2
from scipy.spatial.transform import Rotation as R
import tqdm
import yaml

class CmdLineInterface():

    def __init__(self, **kwargs):
        """
        Initialize the interface with the provided keyword arguments,
        which is assumed to be a dictionary.

        Args:
            **kwargs: Keyword arguments containing the configuration, including:
                - 'input_bag': Path to the input ROS2 bag file.
                - 'output_bag': Path to the output ROS2 bag file, if necessary.
                - 'operation_params': Dictionary containing operation parameters for specific manipulations.
                - 'external_msgs_path_ros2': Path to the directory containing external message definitions.
                - 'operation_to_run': The name of the operation to run, which should be a method of this class.
        """

        # Assign attributes from input arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Create a ROS2 bag wrapper
        self.bag_wrapper = Ros2BagWrapper(self.input_bag, self.external_msgs_path_ros2)

        # Run desired operation
        function = getattr(self, self.operation_to_run)
        function()

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Initialize the CmdLineInterface from a YAML file.

        Args:
            yaml_path (str): Path to the configuration.
        """
        with open(yaml_path, "r") as yaml_file:
            yaml_dict = yaml.safe_load(yaml_file)
            return cls(**yaml_dict)

    # ===============================================================
    # ======================== Operations ===========================
    # ===============================================================

    def downsample(self):
        """ Downsample a ROS2 bag file. """

        # Extract operation specific parameters
        topic_downsample_ratios: dict = self.operation_params['downsample']['topics'].copy()
        include_unmentioned_topics: bool = self.operation_params['downsample']['include_unmentioned_topics']
        self.bag_wrapper.downsample(self.output_bag, topic_downsample_ratios, include_unmentioned_topics)

    def crop(self):
        """ Crop a ROS2 bag file. """

        start_ts: float = self.operation_params['crop']['start_ts']
        end_ts: float = self.operation_params['crop']['end_ts']
        self.bag_wrapper.crop(self.output_bag, start_ts, end_ts)

    def convert_ros2_to_ros1(self):
        """ Convert a ROS2 bag from the bag wrapper into a ROS1 bag. """

        self.bag_wrapper.export_as_ros1(self.output_bag, self.external_msgs_path_ros1)

    def extract_odometry_to_csv(self):
        """ Extract odometry from a ROS2 bag and save in a csv file. """

        topic: str = self.operation_params['extract_odometry_to_csv']['topic']
        output_folder: str = self.operation_params['extract_odometry_to_csv']['output_file']
        add_noise: str = self.operation_params['extract_odometry_to_csv']['add_noise']
        xy_noise_std_per_frame: float = self.operation_params['extract_odometry_to_csv']['xy_noise_std_per_frame']
        z_noise_std_per_frame: str = self.operation_params['extract_odometry_to_csv']['z_noise_std_per_frame']
        shift_position_xy: float = self.operation_params['extract_odometry_to_csv']['shift_position_xy']
        shift_position_z: float = self.operation_params['extract_odometry_to_csv']['shift_position_z']

        odom_data = OdometryData.from_ros2_bag(self.input_bag, topic, CoordinateFrame.NONE)
        if add_noise:
            odom_data.add_folded_guassian_noise_to_position(xy_noise_std_per_frame, z_noise_std_per_frame)
            odom_data.shift_position(shift_position_xy, shift_position_xy, shift_position_z)
        odom_data.to_csv(output_folder)

    def extract_images_to_npy(self):
        """ Extract images from a ROS2 bag and saves them into .npy files. """

        topic: str = self.operation_params['extract_images_to_npy']['topic']
        output_folder: str = self.operation_params['extract_images_to_npy']['output_folder']
        ImageDataInMemory.from_ros2_bag(self.input_bag, topic, output_folder)

    def compare_timestamps_two_image_data(self):
        """ Compare timestamps between two ImageData instances. """

        data0 = ImageDataInMemory.from_npy(self.operation_params['compare_timestamps_two_image_data']['folder_0'])
        data1 = ImageDataInMemory.from_npy(self.operation_params['compare_timestamps_two_image_data']['folder_1'])
        data0.compare_timestamps(data1)

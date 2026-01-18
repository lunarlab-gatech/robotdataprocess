from ..data_types.Data import Data, ROSMsgLibType
from collections import defaultdict
from decimal import Decimal
import glob
import inspect
import numpy as np
from pathlib import Path
import pickle
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag1.writer import Connection
from rosbags.rosbag2 import Reader as Reader2
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import sys
from rosbags.typesys.store import Typestore
import tqdm
from typeguard import typechecked
from typing import Union, Tuple, List, Dict
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Ros2BagWrapper:

    # Define attributes
    bag_path: Path
    external_msgs_path: Union[Path, None]
    typestore2: Typestore
    metadata: dict

    # Only using for exporting to ROS1 bag
    typestore1: Typestore
    ros1_msg_instances: dict
    
    @typechecked
    def __init__(self, bag_path: Union[Path, str], external_msgs_path: Union[Path, str, None]):
        """
        Args:
            input_bag: Path to the input ROS2 bag file.
            external_msgs_path_ros2: Path to the directory containing external message definitions.
        """

        # Ensure both attributes end up as paths (if possible)
        if isinstance(bag_path, str): self.bag_path = Path(bag_path)
        else: self.bag_path = bag_path
        if isinstance(external_msgs_path, str): self.external_msgs_path = Path(external_msgs_path)
        else: self.external_msgs_path = external_msgs_path

        # Create ROS2 Typestore
        self.typestore2 = self._create_typestore_with_external_msgs(Stores.ROS2_HUMBLE, self.external_msgs_path)

        # Load metadata
        self._load_metadata()

    # =========================================================================
    # ============================== Typestores =============================== 
    # =========================================================================  
        
    @staticmethod
    @typechecked
    def _guess_msgtype(path: Path) -> str:
        """
        Guess message type name from path for registration.
        
        Args:
            path (Path): Path to the message file.
        Returns:
            str: The guessed message type name.
        """
        name = path.relative_to(path.parents[2]).with_suffix('')
        if 'msg' not in name.parts:
            name = name.parent / 'msg' / name.name
        return str(name)

    @staticmethod
    @typechecked
    def _create_typestore_with_external_msgs(store_type: Stores, external_msgs_path: Union[Path, None]) -> Typestore:
        """
        Create Typestore with external message types added from the specified path.

        Args:
            store_type (Stores): The rosbags Store type (which defines which ROS version is used).
            external_msgs_path: Path to directory to search for all additional messsages to add.
        Returns:
            Stores: The typestore will all external messages added.
        """

        # Helper method that iterates recusively and finds all .msg files to load
        def register_msgs(path: Path, typestore: Typestore):
            for msg_path in glob.glob(str(path / Path('**') /Path('*.msg')), recursive=True):
                msg_path = Path(msg_path)
                msgdef = msg_path.read_text(encoding='utf-8')
                typestore.register(get_types_from_msg(msgdef, Ros2BagWrapper._guess_msgtype(msg_path)))

        # Initialize the default store
        typestore = get_typestore(store_type)
        
        # Load external message types
        try: register_msgs(external_msgs_path, typestore)
        except: print("No `external_msgs_path` path provided, using no external msgs for ros2 Typestore")
        return typestore
    
    def get_typestore(self) -> Typestore:
        """ Return the ROS2 typestore """
        return self.typestore2

    @staticmethod
    def _create_writer2(path: Union[Path, str]) -> Writer2:
        """
        Create a Writer2 instance with compatibility for different rosbags library versions.
        Older versions (used with Python 3.8) don't support the 'version' parameter.

        Args:
            path (Path | str): Path to the output ROS2 bag file.
        Returns:
            Writer2: A Writer2 instance configured appropriately.
        """
        sig = inspect.signature(Writer2.__init__)
        if 'version' in sig.parameters:
            return Writer2(path, version=9)
        else:
            return Writer2(path)

    # =========================================================================
    # ============================ Hertz Analysis =============================
    # =========================================================================

    @staticmethod
    def create_histogram(data: List, title: str, xlabel: str, ylabel: str, filename: str) -> None:
        """
        Create and save a histogram from the provided data.

        Args:
            data (List): Data to create the histogram from.
            title (str): Title of the histogram.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            filename (str): Filename to save the histogram image.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=100, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.grid(True)
        plt.yscale('log')

        # Save the figure
        if not Path(filename).parent.exists():
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)

    def hertz_analysis(self, topic: str, output_folder: str, expected_msgs: int, max_msgs: int, robot_name: Union[str, None]) -> None:
        """
        Analyze the hertz of various topics in a ROS2 bag file. Will output
        a histogram of the hertz for each pair of consecutive messages
        for the specified topic, and another histogram of the time differences.

        Args:
            topic (str): The topic to analyze.
            output_folder (str): Folder to save the output histograms.
            expected_msgs (int): Expected number of messages in the topic, for progress bar.
            robot_name (str): if 'topic' is '/tf' or '/tf_static', the robot name which then
                choses the transform with a child frame id of '{robot_name}/odom_local'.
            max_msgs (int): The maximum number of messages to use.
        """

        # Open the bag file for reading
        with Reader2(self.bag_path) as reader:
            # Store timestamps
            topic_timestamps = []

            # setup tqdm
            pbar = tqdm.tqdm(total=expected_msgs, desc="Extracting timestamps", unit="frames")
            i = 0

            # Only analyze the specified topic
            connections = [x for x in reader.connections if x.topic == topic]
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                topic = conn.topic
                msg = self.get_typestore().deserialize_cdr(rawdata, conn.msgtype)

                if hasattr(msg, 'transforms'):
                    for transform in msg.transforms:
                        if transform.child_frame_id == robot_name+"/odom_local" \
                        and hasattr(transform, 'header') and hasattr(transform.header, 'stamp'):
                            topic_timestamps.append(Decimal(transform.header.stamp.sec) + Decimal(transform.header.stamp.nanosec) * Decimal(1e-9))
                elif hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    topic_timestamps.append(self.extract_timestamp(msg))
                else:
                    raise ValueError(f"Topic: {topic} does not have a valid header with timestamp.")
                pbar.update(1)
                i += 1
                if i >= max_msgs:
                    break

        # Calculate the differences between consecutive timestamps
        if len(topic_timestamps) < 2:
            raise ValueError(f"Not enough messages in topic {topic} to analyze hertz.")
        hertz_diffs = [topic_timestamps[i] - topic_timestamps[i - 1] for i in range(1, len(topic_timestamps))]
        hertz_values = [1 / diff for diff in hertz_diffs if diff > 0]

        # Output a message if some differences are zero
        num_seq_messages_with_same_timestamps = len([x for x in hertz_diffs if x == 0])
        if num_seq_messages_with_same_timestamps > 0:
            print(f"Warning: Sequential Pairs of timestamps are equivalent {num_seq_messages_with_same_timestamps} times. Excluding from plot...")

        # Sort each of these Lists
        hertz_diffs.sort()
        hertz_values.sort()

        # To remove potentially noisy values, remove first and last 5
        if len(hertz_diffs) > 5:
            hertz_diffs = hertz_diffs[5:-5]
            hertz_values = hertz_values[5:-5]

        # Create histograms for the hertz values and time differences
        robot_str = ""
        if topic == "/tf" or topic == "/tf_static":
            robot_str = robot_name
        self.create_histogram(
            data=hertz_diffs,
            title=f'Time Differences for Topic: {topic}',
            xlabel='Time Difference (seconds)',
            ylabel='Seq. Message Pairs Count (#)',
            filename=Path(output_folder, f'{robot_str}_{topic.replace("/", "_")}_DIFFS.png')
        )
        self.create_histogram(
            data=hertz_values,
            title=f'Hertz Analysis for Topic: {topic}',
            xlabel='Hertz (Hz)',
            ylabel='Seq. Message Pairs Count (#)',
            filename=Path(output_folder, f'{robot_str}_{topic.replace("/", "_")}_HERTZ.png')
        )

    # =========================================================================
    # ============================ View IMU Data ==============================
    # =========================================================================

    def view_imu_data(self, topic: str, output_folder: str, expected_msgs: Union[int, None], data_range: Union[Tuple[int], None]):
        """
        Plot the linear acceleration, angular velocity, and orientation data of an IMU topic in a ROS2 bag.

        Args:
            topic (str): The topic to read the IMU data from; must be 'sensor_msgs/msg/Imu'.
            output_folder (str): The folder to save plotted IMU data.
            expected_msgs: Expected number of messages in the topic, for progress bar.
            data_range: The range of data to plot.
        """

        # Create the output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Check if we've already parsed the bag into a .npy file
        npy_path = Path(output_folder) / "imu_data.npy"
        if npy_path.exists():

            # If so, load the cached data
            data = np.load(npy_path, allow_pickle=True).item()
            timestamps = data['timestamps']
            lin_accel = data['lin_accel']
            ang_vel = data['ang_vel']
            orientation = data['orientation']

        else: # Otherwise, read the bag

            # Initialize Lists
            lin_accel = {'x': [], 'y': [], 'z': []}
            ang_vel = {'x': [], 'y': [], 'z': []}
            orientation = {'x': [], 'y': [], 'z': [], 'w': []}
            timestamps = []

            # Open the bag file for reading
            with Reader2(self.bag_path) as reader:

                # setup tqdm
                pbar = tqdm.tqdm(total=expected_msgs, desc="Extracting IMU Data", unit=" frames")

                # Only analyze the specified topic
                connections = [x for x in reader.connections if x.topic == topic]
                if len(connections) != 1:
                    raise ValueError("Input topic doesn't exist in bag!")

                # Iterate through messages
                for conn, timestamp, rawdata in reader.messages(connections=connections):
                    msg = self.get_typestore().deserialize_cdr(rawdata, conn.msgtype)

                    # Timestamp
                    timestamps.append(Decimal(msg.header.stamp.sec) + Decimal(msg.header.stamp.nanosec) * Decimal(1e-9))

                    # Linear acceleration
                    lin_accel['x'].append(msg.linear_acceleration.x)
                    lin_accel['y'].append(msg.linear_acceleration.y)
                    lin_accel['z'].append(msg.linear_acceleration.z)

                    # Angular velocity
                    ang_vel['x'].append(msg.angular_velocity.x)
                    ang_vel['y'].append(msg.angular_velocity.y)
                    ang_vel['z'].append(msg.angular_velocity.z)

                    # Orientation
                    orientation['x'].append(msg.orientation.x)
                    orientation['y'].append(msg.orientation.y)
                    orientation['z'].append(msg.orientation.z)
                    orientation['w'].append(msg.orientation.w)

                    pbar.update(1)

            # Convert quaternions into euler angles
            quaternions = np.array([ orientation['x'], orientation['y'],
                orientation['z'], orientation['w']]).T
            euler_angles = R.from_quat(quaternions).as_euler('xyz', degrees=True)
            orientation = {'roll': euler_angles[:, 0], 'pitch': euler_angles[:, 1], 'yaw': euler_angles[:, 2]}

            # Save extracted data into a .npy file
            np.save(Path(output_folder) / "imu_data.npy", {
                'timestamps': timestamps,
                'lin_accel': lin_accel,
                'ang_vel': ang_vel,
                'orientation': orientation,
            })

        def multi_list_plotter(data_dict: dict, timestamps: list, data_range: Tuple[int],
                               title: str, ylabel: str, filename: str):
            """
            Helper function that plots each list in a dictionary in its own subplot,
            all in a single matplotlib figure.
            """

            num_plots = len(data_dict)
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)

            # Make it a list always
            if num_plots == 1:
                axes = [axes]

            shift = timestamps[0]
            for i in range(0, len(timestamps)):
                timestamps[i] -= shift

            if data_range is not None:
                timestamps = timestamps[data_range[0]:data_range[1]]

            for ax, (key, values) in zip(axes, data_dict.items()):
                if data_range is not None:
                    values = values[data_range[0]:data_range[1]]
                ax.plot(timestamps, values, label=key)
                ax.set_ylabel(f"{key} - ({ylabel})")
                ax.grid(True)
                ax.legend(loc='upper right')

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{title} vs Time", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(str(Path(output_folder) / filename))
            plt.close()

        # Create all plots
        multi_list_plotter(lin_accel, timestamps, data_range, "Linear Acceleration", "m/s^2", "linear_acceleration.png")
        multi_list_plotter(ang_vel, timestamps, data_range, "Angular Velocity", "rad/s", "angular_velocity.png")
        multi_list_plotter(orientation, timestamps, data_range, "Orientation", "degrees", "orientation.png")

    # =========================================================================
    # ============================= Downsampling ==============================
    # =========================================================================

    def downsample(self, output_bag: Union[Path, str], topic_downsample_ratios: dict, include_unmentioned_topics: bool):
        """
        Downsample a ROS2 bag file by downsampling the frequency of specified topics. Additionally
        includes or excludes unmentioned topics based on `include_unmentioned_topics` parameter, which
        can be used to prune topics.

        If the number of msgs on a topic multiplied by the downsample ratio is not an integer
        (ex. 696 * 0.1 = 69.6), then the number of messages will be the nearest integer of the
        calculated float (ex. np.round(69.6) == 70).

        Args:
            output_bag (Path | str): Path to the output ROS2 bag file.
            topic_downsample_ratios (dict): A dictionary mapping topic names to their downsample ratios.
            include_unmentioned_topics (bool): Boolean indicating whether to include topics not
                mentioned in the downsample list in the output bag.
        """

        # Convert to pathlib paths
        output_bag = Path(output_bag)

        if include_unmentioned_topics:
            raise NotImplementedError('Logic for this isnt fully tested')

        # Ensure we aren't overwriting an existing output bag
        if output_bag.exists():
            raise AssertionError("Delete Output Directory first!")

        # Check for valid input downsample ratios
        for topic in topic_downsample_ratios:
            ratio = topic_downsample_ratios[topic]
            if ratio <= 0 or ratio > 1:
                raise ValueError("Downsample rates must be >0 and <= 1")

        # Get number of messages on each of the topics
        topic_counts = {}
        for topic_info in self.metadata.get('rosbag2_bagfile_information').get('topics_with_message_count'):
            topic_name = topic_info['topic_metadata']['name']
            message_count = topic_info['message_count']
            topic_counts[topic_name] = message_count

        # Calculate which messages we want to save for each topic
        indicies_to_save = {}
        for topic in topic_counts:
            if topic in topic_downsample_ratios:
                desired_final_count = int(np.round(topic_counts[topic] * topic_downsample_ratios[topic]))
                indicies_to_save[topic] = set(np.linspace(0, topic_counts[topic] - 1, num=desired_final_count, dtype=int).tolist())

        # Open the bag
        with Reader2(self.bag_path) as reader:
            connections = reader.connections

            # Create a writer with only the specified topics
            with Ros2BagWrapper._create_writer2(output_bag) as writer:
                conn_map = {}
                for conn in connections:
                    if (conn.topic in topic_downsample_ratios) or include_unmentioned_topics:
                        conn_map[conn.topic] = writer.add_connection(
                            topic=conn.topic,
                            msgtype=conn.msgtype,
                            msgdef=conn.msgdef,
                            typestore=self.get_typestore(),
                            serialization_format='cdr',
                            offered_qos_profiles=conn.ext.offered_qos_profiles
                        )

                # Initialize counters for each topic
                topic_counters = defaultdict(int)
                for topic in topic_downsample_ratios:
                    topic_counters[topic] = 0

                # Setup progress bars
                pbarC = tqdm.tqdm(total=None, desc="Reading Requested Messages", unit=" messages")
                pbarW = tqdm.tqdm(total=None, desc="Writing Messages", unit=" messages")

                # Iterate through reader only through desired topics
                if include_unmentioned_topics:
                    connections = reader.connections
                else:
                    connections = [x for x in reader.connections if x.topic in conn_map]

                # Interate through messages in the bag
                for conn, timestamp, rawdata in reader.messages(connections=connections):
                    pbarC.update(1)

                    # See if we need to save this message or cut it
                    if topic_counters[conn.topic] in indicies_to_save[conn.topic]:
                        pbarW.update(1)
                        writer.write(conn_map[conn.topic], timestamp, rawdata)

                    # Increment the counter for this topic
                    topic_counters[conn.topic] += 1

                # Close the progress bars
                pbarC.close()
                pbarW.close()

        print(f"Downsampling complete. Output bag saved to {output_bag}.")

    # =========================================================================
    # =============================== Metadata ================================ 
    # =========================================================================  

    def _load_metadata(self) -> None:
        """
        Open the metadata.yaml file and save it for use later.

        Raises:
            FileNotFoundError: If the metadata.yaml file doesn't exist.
        """

        # Make sure the metadata.yaml file exists
        metadata_path = self.bag_path / 'metadata.yaml'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Unable to find metadata.yaml in ROS2 bag: {metadata_path}")
        
        # Load the file with yaml
        with open(metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)

    def get_topic_count(self, topic_name: str) -> int:
        """
        Get the number of messages on the specified topic.

        Args:
            topic_name (str): The name of the topic.
        Returns:
            int: Number of messages.
        Raises:
            ValueError: If topic_name doesn't exist in the bag.
        """

        # Iterate through each topic until we find the one we want
        for topic_info in self.metadata.get('rosbag2_bagfile_information').get('topics_with_message_count'):
            curr_name = topic_info['topic_metadata']['name']
            if topic_name == curr_name:
                return topic_info['message_count']
            
        # If we get here, this topic doesn't exist in the bag
        raise ValueError(f"Topic '{topic_name}' doesn't exist in the current bag!")
    
    # =========================================================================
    # ============================== Utilities ================================ 
    # =========================================================================  

    @staticmethod
    def extract_timestamp(msg: object) -> Decimal:
        """
        Helper method that extracts a timestamp from a ROS message.

        Args:
            msg (object): The message to extract the timestamp from.
        Returns:
            Decimal: Timestamp of message with maximum precision to the nanosecond.
        """

        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            return Decimal(msg.header.stamp.sec) + Decimal(msg.header.stamp.nanosec) * Decimal("1e-9")
        else:
            raise ValueError(f"Messages does not have a valid header with timestamp: {msg}")
        
    # =========================================================================
    # ============================== Cropping ================================= 
    # =========================================================================  

    def crop(self, output_path: Union[Path, str], start_ts: float, end_ts: float) -> None:
        """
        Crop this ROS2 bag file to only include messages within the specified time range. Note
        that this function uses the timestamps of when the messages were written to the bag,
        not the timestamps in the headers of the messages themselves.

        Args:
            output_path (Path | str): Path to save the output ROS2 bag file.
            start_ts (float): Start timestamp in seconds for cropping.
            end_ts (float): End timestamp in seconds for cropping.
        """

        # Make sure we aren't overwriting an existing output bag
        output_bag = Path(output_path)
        if output_bag.exists():
            raise AssertionError("Delete Output Directory first!")

        # Open the current bag
        with Reader2(self.bag_path) as reader:
            connections = reader.connections
            
            # Setup the bag writer
            with Ros2BagWrapper._create_writer2(output_bag) as writer:
                conn_map = {}
                for conn in connections:
                    conn_map[conn.topic] = writer.add_connection(
                        topic=conn.topic,
                        msgtype=conn.msgtype,
                        msgdef=conn.msgdef,
                        typestore=self.get_typestore(),
                        serialization_format='cdr',
                        offered_qos_profiles=conn.ext.offered_qos_profiles
                    )

                # setup tqdm 
                pbarW = tqdm.tqdm(total=None, desc="Writing messages", unit=" messages")

                # Iternate through messages in the bag
                for conn, timestamp, rawdata in reader.messages(start=start_ts * 1e9, stop=end_ts * 1e9):
                    writer.write(conn_map[conn.topic], timestamp, rawdata)
                    pbarW.update(1)
        
    # =========================================================================
    # ======================== Write Data to a Rosbag =========================
    # ========================================================================= 
    
    @staticmethod
    @typechecked
    def write_data_to_rosbag(bag_path: Union[Path, str], data_list: List[Data], 
                             data_topics: List[str], data_msg_type: List[Union[str, None]],
                             external_msgs_path: Union[Path, str, None]):
        """
        This helper method writes a new bag with the data in the provided Data classes.

        Args:
            bag_path (Path | str): The path to write the output bag.
            data_list (List[Data]): A list of all Data objects whose contents should
                be written into the bag.
            data_topics (List[str]): The topics under which the corresponding Data class's
                data should be saved.
            data_msg_type (List[str]): If supported, the corresponding message type that
                the data class data should be written into.
            external_msgs_path (Path | str | None): Path to the directory containing 
                external message definitions.
        """

        # Create the typestore
        typestore = Ros2BagWrapper._create_typestore_with_external_msgs(Stores.ROS2_HUMBLE, external_msgs_path)

        # Open a ROS2 Humble bag for writing
        with Ros2BagWrapper._create_writer2(str(bag_path)) as writer:

            # For each data class
            for i in range(0, len(data_list)):
                data = data_list[i]
                topic = data_topics[i]

                # Add the new connection
                if data_msg_type[i] is not None: msgtype = data.get_ros_msg_type(ROSMsgLibType.ROSBAGS, data_msg_type[i])
                else: msgtype = data.get_ros_msg_type(ROSMsgLibType.ROSBAGS)
                connection = writer.add_connection(topic, msgtype, typestore=typestore)

                # Setup a tqdm bar
                pbar = tqdm.tqdm(total=data.len(), desc=f"Writing {topic}", unit=" messages")

                # Write each of the data entries
                for j in range(0, data.len()):
                    if data_msg_type[i] is not None: msg = data.get_ros_msg(ROSMsgLibType.ROSBAGS, j, data_msg_type[i])
                    else: msg = data.get_ros_msg(ROSMsgLibType.ROSBAGS,j)
                    timestamp = int(Ros2BagWrapper.extract_timestamp(msg) * Decimal('1e9'))
                    writer.write(connection, timestamp, typestore.serialize_cdr(msg, msgtype))
                    pbar.update(1)

    # =========================================================================
    # ============================= Conversions ===============================
    # ========================================================================= 

    # Defines msg_type mappings from ROS2 to ROS1
    msg_mapping_ros2_to_ros1 = {
        "builtin_interfaces/msg/Time": "builtin_interfaces/msg/Time",
        "geometry_msgs/msg/Point": "geometry_msgs/msg/Point",
        "geometry_msgs/msg/Pose": "geometry_msgs/msg/Pose",
        "geometry_msgs/msg/PoseStamped": "geometry_msgs/msg/PoseStamped",
        "geometry_msgs/msg/PoseWithCovariance": "geometry_msgs/msg/PoseWithCovariance",
        "geometry_msgs/msg/Quaternion": "geometry_msgs/msg/Quaternion",
        "geometry_msgs/msg/Transform": "geometry_msgs/msg/Transform",
        "geometry_msgs/msg/TransformStamped": "geometry_msgs/msg/TransformStamped",
        "geometry_msgs/msg/Twist": "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistWithCovariance": "geometry_msgs/msg/TwistWithCovariance",
        "geometry_msgs/msg/Vector3": "geometry_msgs/msg/Vector3",
        "nav_msgs/msg/Odometry": "nav_msgs/msg/Odometry",
        "nav_msgs/msg/Path": "nav_msgs/msg/Path",
        "rosgraph_msgs/msg/Clock": "rosgraph_msgs/msg/Clock",
        "sensor_msgs/msg/Image": "sensor_msgs/msg/Image",
        "sensor_msgs/msg/Imu": "sensor_msgs/msg/Imu",
        "sensor_msgs/msg/CameraInfo": "sensor_msgs/msg/CameraInfo",
        "sensor_msgs/msg/PointCloud2": "sensor_msgs/msg/PointCloud2",
        "sensor_msgs/msg/PointField": "sensor_msgs/msg/PointField",
        "sensor_msgs/msg/RegionOfInterest": "sensor_msgs/msg/RegionOfInterest",
        "std_msgs/msg/Header": "std_msgs/msg/Header",
        "tf2_msgs/msg/TFMessage": "tf2_msgs/msg/TFMessage"
    }

    def _get_mapping_2to1(self, msg_type: str) -> dict:
        """
        Given the msg_type string, return a mapping dictionary that describes how to map 
        values from a ros2 message to a ros1 message. Only supports a limited set of 
        message types currently, but can be extended to support more.

        Note that defining mappings of different depths is not allowed, and will
        break list mapping functionality (i.e. no "header.item1.child1": "header.child1").

        Args:
            msg_type (str): The message type string, e.g., "sensor_msgs/msg/Image".
        Returns:
            dict: A mapping dictionary where:
                keys (str): A string defining the attribute in the ros2 message.
                    Thus, a key of "header.stamp" would map to ros2_msg.header.stamp.
                values (str or Tuple): A string defining the attribute in the ros1 message,
                    OR a Tuple where the first element is the attribute in the ros1 message and
                    the second element is a mapping dictionary for mapping attributes within 
                    child message types.
        """

        map = None
        if msg_type == "builtin_interfaces/msg/Time":
            map = {
                "sec": "sec",
                "nanosec": "nanosec"
            }
        elif msg_type == "geometry_msgs/msg/Point":
            map = {
                "x": "x",
                "y": "y",
                "z": "z"
            }
        elif msg_type == "geometry_msgs/msg/Pose":
            map = {
                "position": ("position", self._get_mapping_2to1("geometry_msgs/msg/Point")),
                "orientation": ("orientation", self._get_mapping_2to1("geometry_msgs/msg/Quaternion"))
            }
        elif msg_type == "geometry_msgs/msg/PoseStamped":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "pose": ("pose", self._get_mapping_2to1("geometry_msgs/msg/Pose"))
            }
        elif msg_type == "geometry_msgs/msg/PoseWithCovariance":
            map = {
                "pose": ("pose", self._get_mapping_2to1("geometry_msgs/msg/Pose")),
                "covariance": "covariance"
            }
        elif msg_type == "geometry_msgs/msg/Quaternion":
            map = {
                "x": "x",
                "y": "y",
                "z": "z",
                "w": "w"
            }
        elif msg_type == "geometry_msgs/msg/Transform":
            map = {
                "translation": ("translation", self._get_mapping_2to1("geometry_msgs/msg/Vector3")),
                "rotation": ("rotation", self._get_mapping_2to1("geometry_msgs/msg/Quaternion"))
            }
        elif msg_type == "geometry_msgs/msg/TransformStamped":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "child_frame_id": "child_frame_id",
                "transform": ("transform", self._get_mapping_2to1("geometry_msgs/msg/Transform")),
            }
        elif msg_type == "geometry_msgs/msg/Twist":
            map = {
                "linear": ("linear", self._get_mapping_2to1("geometry_msgs/msg/Vector3")),
                "angular": ("angular", self._get_mapping_2to1("geometry_msgs/msg/Vector3"))
            }
        elif msg_type == "geometry_msgs/msg/TwistWithCovariance":
            map = {
                "twist": ("twist", self._get_mapping_2to1("geometry_msgs/msg/Twist")),
                "covariance": "covariance"          
            }
        elif msg_type == "geometry_msgs/msg/Vector3":
            map = {
                "x": "x",
                "y": "y",
                "z": "z"
            }
        elif msg_type == "nav_msgs/msg/Odometry":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "child_frame_id": "child_frame_id",
                "pose": ("pose", self._get_mapping_2to1("geometry_msgs/msg/PoseWithCovariance")),
                "twist": ("twist", self._get_mapping_2to1("geometry_msgs/msg/TwistWithCovariance"))
            }
        elif msg_type == "nav_msgs/msg/Path":
            map = { # TODO: This is broken, as it can't handle if non-final mapping is a list
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "poses": ("poses", self._get_mapping_2to1("geometry_msgs/msg/PoseStamped"))
            }
        elif msg_type == "rosgraph_msgs/msg/Clock":
            map = {
                "clock": ("clock", self._get_mapping_2to1("builtin_interfaces/msg/Time"))
            }
        elif msg_type == "sensor_msgs/msg/Image":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "height": "height",
                "width": "width",
                "encoding": "encoding",
                "is_bigendian": "is_bigendian",
                "step": "step",
                "data": "data"
            }
        elif msg_type == "sensor_msgs/msg/Imu":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "orientation": ("orientation", self._get_mapping_2to1("geometry_msgs/msg/Quaternion")),
                "orientation_covariance": "orientation_covariance",
                "angular_velocity": ("angular_velocity", self._get_mapping_2to1("geometry_msgs/msg/Vector3")),
                "angular_velocity_covariance": "angular_velocity_covariance",
                "linear_acceleration": ("linear_acceleration", self._get_mapping_2to1("geometry_msgs/msg/Vector3")),
                "linear_acceleration_covariance": "linear_acceleration_covariance"
            }
        elif msg_type == "sensor_msgs/msg/CameraInfo":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "height": "height",
                "width": "width",
                "distortion_model": "distortion_model",
                "d": "D",
                "k": "K",
                "r": "R",
                "p": "P",
                "binning_x": "binning_x",
                "binning_y": "binning_y",
                "roi": ("roi", self._get_mapping_2to1("sensor_msgs/msg/RegionOfInterest"))
            }
        elif msg_type == "sensor_msgs/msg/PointCloud2":
            map = {
                "header": ("header", self._get_mapping_2to1("std_msgs/msg/Header")),
                "height": "height",
                "width": "width",
                "fields": ("fields", self._get_mapping_2to1("sensor_msgs/msg/PointField")),
                "is_bigendian": "is_bigendian",
                "point_step": "point_step",
                "row_step": "row_step",
                "data": "data",
                "is_dense": "is_dense"
            }
        elif msg_type == "sensor_msgs/msg/PointField":
            map = {
                "name": "name",
                "offset": "offset",
                "datatype": "datatype",
                "count": "count"
            }
        elif msg_type == "sensor_msgs/msg/RegionOfInterest":
            map = {
                "x_offset": "x_offset",
                "y_offset": "y_offset",
                "height": "height",
                "width": "width",
                "do_rectify": "do_rectify"
            }
        elif msg_type == "std_msgs/msg/Header":
            map = {
                "stamp": ("stamp", self._get_mapping_2to1("builtin_interfaces/msg/Time")),
                "frame_id": "frame_id"
                # Note: ROS1 msg has a 'seq' value, which we define no mapping for.
                # Thus, default value in self._get_ros1_msg_instance is used.
            }
        elif msg_type == "tf2_msgs/msg/TFMessage":
            map = { # TODO: This is broken, as it can't handle if non-final mapping is a list
                "transforms": ("transforms", self._get_mapping_2to1("geometry_msgs/msg/TransformStamped"))
            }
        else:
            raise ValueError(f"Currently unsupported msg_type: {msg_type}")
            
        return map
    
    def _invert_map(self, map: dict) -> dict:
        """
        Given an input mapping of values from different ROS verions, flip it so that 
        the keys are the values and the values are the keys. This equates to converting
        a mapping from ROS2 to ROS1 into a mapping from ROS1 to ROS2. Handles special
        cases where the value is a Tuple instead of a string.

        Args:
            map (dict): Dictionary returned by _get_mapping_2to1() or _get_mapping_1to2()

        Returns:
            dict: Dictionary with an inverse mapping from before.
        """

        inverted_map = {}
        for k, v in map.items():
            if isinstance(v, tuple) and len(v) == 2:
                new_key = v[0]
                new_value = (k, self._invert_map(v[1]))
            else:
                new_key = v
                new_value = k
            inverted_map[new_key] = new_value
        return inverted_map
    
    def _get_mapping_1to2(self, msg_type: str) -> dict:
        """
        Same as _get_mapping_2to1(), but maps values from a ros1 message to a ros2 message.

        Args:
            msg_type (str): The message type string, e.g., "sensor_msgs/msg/Image".
        Returns:
            dict: A mapping dictionary where:
                keys (str): A string defining the attribute in the ros2 message.
                    Thus, a key of "header.stamp" would map to ros2_msg.header.stamp.
                values (str or Tuple): A string defining the attribute in the ros1 message,
                    OR a Tuple where the first element is the attribute in the ros1 message and
                    the second element is a mapping dictionary for mapping attributes within 
                    child message types.
        """

        map = self._get_mapping_2to1(msg_type)
        return self._invert_map(map)
        
    def _solve_for_all_mappings(self, map: Union[Dict[str, Union[str, Tuple]], None], msg_type: Union[str, None]) -> dict:
        """
        Helper method that will take the provided map and recursively solve for all mappings
        that are not yet resolved. Additionally, if a map isn't provided, it will generate
        an initial map from the provided msg_type.

        Args:
            map (dict | None): A mapping dictionary where:
                keys (List[str]): A list defining the attribute in the ros2 message.
                    Thus, a key of "header.stamp" would map to ros2_msg.header.stamp.
                values (List[str | Tuple]): A list defining the attribute in the ros1 message,
                    OR a Tuple where the first element is the attribute in the ros1 message and
                    the second element is a mapping dictionary for mapping attributes within 
                    child message types.
              If None, then 'msg_type' is required to generate an initial map.
            msg_type (str | None): The message type string, e.g., "sensor_msgs/msg/Image".
                If map is None, an initial map will be generated from the msg_type.
        
        Returns:
            dict: A mapping dictionary where:
                keys (list of str): A list defining the attribute in the ros2 message.
                    Thus, a key of "header.stamp" would map to ros2_msg.header.stamp.
                values (list of str): A list defining the attribute in the ros1 message. Note
                that this can no longer be a Tuple, as all mappings have been resolved.
        """
        
        # Check input parameters
        if map is None and msg_type is not None:
            # If no map is provided, we need to generate it from the msg_type
            map = self._get_mapping_2to1(msg_type)
        elif map is None and msg_type is None:
            raise ValueError("No map provided and no msg_type specified to generate initial map.")
        elif map is not None and msg_type is not None:
            print("WARNING: msg_type provided, but map is already defined. Using provided map and ignoring msg_type.")

        # Iterate through each key in the map to check for unresolved mappings
        for key in map.copy():
            # There are unresolved mappings, so we need to solve them
            if type(map[key]) is tuple:

                # Extract value and child map
                value = map[key][0]
                child_map = map[key][1]

                # Solve recursively to only get values with type "list" in the child map
                child_map = self._solve_for_all_mappings(child_map, None)

                # Now, take child map and bring it into the current map
                for child_key in child_map:
                    new_key = '.'.join(key.split('.') + child_key.split('.'))
                    new_value = '.'.join(value.split('.') + child_map[child_key].split('.'))
                    map[new_key] = new_value
                
                # Remove the old key
                del map[key]
        return map
    
    def _get_default_param(self, type_str: str, ros2_msg_attr: object):
        """
        This method will inspect the type provided in type_str and
        return a default value to use in the instanciated ROS1 msg.
        If no mapping is defined for this value from the ROS2 msg, 
        this will be the final value.

        Args:
            type_str (str): The attribute type to get a default value
                for.
            ros2_msg_attr (object): The ROS2 object that corresponds
                to this parameter, if it exists. Is used to determine
                list sizes in _get_ros1_msg_instance().
        """

        if type_str == "<class 'inspect._empty'>" or type_str == 'ClassVar[str]':
            pass
        elif type_str == "int":
            return 0
        elif type_str == "str":
            return ""
        elif type_str == "bool":
            return False
        elif type_str == "float":
            return 0.0
        elif "np.ndarray" in type_str:
            return np.array([])
        elif "msg" in type_str:
            return self._get_ros1_msg_instance("/".join(type_str.split('__')), ros2_msg_attr)
        else:
            raise NotImplementedError(f"Unspecified default value: {type_str}")

    def _get_ros1_msg_instance(self, msg_type: str, ros2_msg: Union[object, None]) -> object:
        """
        Get a ROS1 message instance with all values filled with defaults.
        This allows us to easily add attributes later on, instead of needing
        all of the attributes ready to go before initialization.

        Args:
            msg_type (str): The type of message to initialize a ROS1 msg class for.
            ros2_msg (object | None): The ros2_msg that corresponds to the ros1 msg instance
                we want to build. Only used if message contains a list, so that the 
                ros1 msg can have a list of the same size. Set to none if it doesn't exist.

        Returns:
            object: The initialized ROS1 class message.
        """

        # If we've already made an instance, return it
        try:
            # Doesn't work for messages with lists, as list size might vary
            if msg_type != 'tf2_msgs/msg/TFMessage':
                return pickle.loads(self.ros1_msg_instances[msg_type])
        except:
            pass

        # Initialize the ros1_msg class
        ros1_msg_class = self.typestore1.types[self.msg_mapping_ros2_to_ros1[msg_type]]

        # Get the ros1 to ros2 mapping
        map1to2 = self._get_mapping_1to2(msg_type)

        # Get the signature of the __init__ method
        sig = inspect.signature(ros1_msg_class.__init__)

        # Set default values
        initial_parameters = []
        for name, param in sig.parameters.items():

            # Skip 'self' as it needs no default value
            if name == 'self':
                continue

            # Get the type of this parameter
            type_str = str(param.annotation)

            # Skip ClassVar parameters as they are class-level attributes, not instance parameters
            if 'ClassVar' in type_str:
                continue

            # Get the ros2 object for this parameter
            try:
                name_ros2 = map1to2[name]
                if type(name_ros2) == tuple:
                    name_ros2 = name_ros2[0]
                ros2_msg_attr = getattr(ros2_msg, name_ros2)
            except: # No matching ros2 mapping
                ros2_msg_attr = None

            # If it is a list, add as many default values to list as needed
            if "list" in type_str:

                # Make sure there aren't nested lists
                type_str_child = type_str[5:-1]
                if "list" in type_str_child:
                    raise NotImplementedError("ROS Messages with nested lists are not supported.")

                # Find the number of items in the corresponding list in the ros2 message
                if ros2_msg_attr is not None:
                    list_size = len(ros2_msg_attr)
                else:
                    # There is no matching value in ros2 message to pass.
                    # Thus, just append an empty list
                    initial_parameters.append([])

                # Add the child item as many times as necessary
                list = []
                for i in range(0, list_size):
                    list.append(self._get_default_param(type_str_child, ros2_msg_attr))
                initial_parameters.append(list)
            
            # Otherwise, just add the default value once
            else: 
                initial_parameters.append(self._get_default_param(type_str, ros2_msg_attr))

        # Initialize class with dummy values
        instance = ros1_msg_class(*initial_parameters)

        # If possible (msg doesn't have lists), save it so we can reuse it next time
        if msg_type != "tf2_msgs/msg/TFMessage":
            # Dump it to a pickle object for deep copying later, but do it now for speed
            self.ros1_msg_instances[msg_type] = pickle.dumps(instance)
        return instance

    def _iterate_and_assign(self, ros2_attr_list: List[str], ros1_attr_list: List[str], 
                           ros2_obj: object, ros1_obj: object, msg_type: str):
        """
        Given attribute lists and corresponding ROS msg objects, isolate the 
        individual final attributes and assign the ros2 value to the ros1 
        value. Handles edge cases where attributes are lists.

        Args:
            ros2_attr_list (list): list of strings defining path of attributes
                from current ros2_obj to final attribute to use as new value
                for corresponding ros1_boj final attribute.
            ros1_attr_list (list): list of strings defining path of attributes
                from current ros1_obj to final attribute to assign with a new
                value from the ros2_obj.
            ros2_obj (object): The ROS2 object that contains the attribute to use.
            ros1_obj (object): The ROS1 object that contains the attribute to assign.
            msg_type (str): Holds the msg_type when this method was called the
                first time (non-recursively). Only used for an error message.

        """
        # See if either attr_list has a list object in it
        list_found_2, list_found_1 = False, False

        # Iterate down to the base attribute in each message
        depth_ros2, depth_ros1 = 0, 0
        while len(ros2_attr_list) > 1 and not list_found_2:
            ros2_obj = getattr(ros2_obj, ros2_attr_list[0])
            if type(ros2_obj) == list:
                list_found_2 = True
            ros2_attr_list = ros2_attr_list[1:]
            depth_ros2 += 1

        while len(ros1_attr_list) > 1 and not list_found_1:
            ros1_obj = getattr(ros1_obj, ros1_attr_list[0])
            if type(ros1_obj) == list:
                list_found_1 = True
            ros1_attr_list = ros1_attr_list[1:]
            depth_ros1 += 1

        # If a list was found, see if we can still assign
        if list_found_2 or list_found_1:

            # Double check both objects are currently lists
            if type(ros1_obj) != list or type(ros2_obj) != list:
                raise RuntimeError("Attempting to assign list object to single object (or visa versa). Double check mappings in _get_mapping_2to1().")

            # Double check list objects are at same depth
            if depth_ros1 != depth_ros2:
                raise RuntimeError("List attributes found at different depths; this should not happen. Make sure there are no map assignments with varying depth in _get_mapping_2to1().")
            
            # Call this method recursively for each time we need to set the object
            for i in range(0, len(ros1_obj)):
                self._iterate_and_assign(ros2_attr_list, ros1_attr_list, ros2_obj[i], ros1_obj[i], msg_type)

        # Otherwise, assign the value as normal
        else:
            # Make sure the attribute we're about to set has a default value (and thus exists)
            if hasattr(ros1_obj, ros1_attr_list[0]):
                # If so, set it
                setattr(ros1_obj, ros1_attr_list[0], getattr(ros2_obj, ros2_attr_list[0]))
            else:
                raise Exception(f"Attribute '{ros1_attr_list[0]}' to set somewhere in {msg_type} doesn't exist! Double check values in self._get_mapping_2to1().")
        
    def _map_ros2_to_ros1(self, ros2_msg: object, msg_type: str) -> object:
        """
        Map a ROS2 message to a ROS1 message based on the provided message type.

        Args:
            ros2_msg (object): The ROS2 message object to be mapped.
            msg_type (str): The message type string, e.g., "sensor_msgs/msg/Image".

        Returns:
            object: The mapped ROS1 message object.
        """

        # Get ros1 message class
        ros1_msg = self._get_ros1_msg_instance(msg_type, ros2_msg)

        # Map attributes from ros2_msg to ros1_msg
        for ros2_attr_str, ros1_attr_str in self._solve_for_all_mappings(None, msg_type).items():
            ros2_attr_list = ros2_attr_str.split('.')
            ros1_attr_list = ros1_attr_str.split('.')

            # Map each message to temporary objects
            ros2_obj = ros2_msg
            ros1_obj = ros1_msg

            # Iterate through this attribute and any child messages and assign values
            self._iterate_and_assign(ros2_attr_list, ros1_attr_list, ros2_obj, ros1_obj, msg_type)
        
        return ros1_msg

    def export_as_ros1(self, output_path: Union[Path, str], external_msg_path_ros1: Union[Path, str]):
        """
        Export the ROS2 bag file this class is wrapping into a ROS1 bag file.

        Args:
            output_path (Path | str): Path to the output ROS1 bag file.
            external_msgs_path_ros2 (Path | str): Path to the directory containing external 
                message definitions.

        Warning:
            Known bug that nav_msgs/Path messages are not properly written to the ROS1 bag, as
            well as tf2_msgs/TFMessage messages.
        """

        # Convert to pathlib paths
        output_bag = Path(output_path)

        # Create a Typestore for ROS1 Noetic & holder of ros1 msg instances
        self.typestore1 = self._create_typestore_with_external_msgs(Stores.ROS1_NOETIC, external_msg_path_ros1)
        self.ros1_msg_instances = {}

        # Make sure we aren't overwriting an existing output bag
        if output_bag.exists():
            raise AssertionError("Delete Output Directory first!")

        with Reader2(self.bag_path) as reader:
            connections = reader.connections
            
            # Setup the bag writer
            with Writer1(output_bag) as writer:
                conn_map = {}
                for conn in connections:
                    # Determine latching behavior (if possible)
                    latching = None
                    if len(conn.ext.offered_qos_profiles) == 1:
                        if conn.ext.offered_qos_profiles[0].durability.value == 2: latching = 0 # VOLATILE
                        elif conn.ext.offered_qos_profiles[0].durability.value == 1: latching = 1 # TRANSIENT_LOCAL
                    else: pass

                    # Only write supported types
                    msg_def = None
                    try: 
                        msg_def = self.typestore1.types[self.msg_mapping_ros2_to_ros1[conn.msgtype]]
                    except:
                        print(f"No mapping currently provided for {conn.msgtype}, skipping...")
                        continue

                    # Add the connection
                    new_conn: Connection = writer.add_connection(
                        topic=conn.topic,
                        msgtype=self.msg_mapping_ros2_to_ros1[conn.msgtype],
                        msgdef=msg_def,
                        typestore=self.typestore1,
                        md5sum=None,
                        callerid=None,
                        latching=latching
                    )
                    conn_map[conn.topic] = new_conn

                # setup tqdm 
                pbarW = tqdm.tqdm(total=None, desc="Writing messages", unit=" messages")

                # Iternate through messages in the bag
                for conn, timestamp, rawdata in reader.messages():
                    # Only writes messages that had a connection successfully created
                    if conn.topic in list(conn_map.keys()):
                        # Deserialize the ROS2 message
                        ros2_msg = self.get_typestore().deserialize_cdr(rawdata, conn.msgtype)

                        # Map the ROS2 message to a ROS1 message
                        ros1_msg = self._map_ros2_to_ros1(ros2_msg, conn.msgtype)

                        # Serialize the ROS1 message
                        rawdata = self.typestore1.serialize_ros1(ros1_msg, self.msg_mapping_ros2_to_ros1[conn.msgtype])

                        # Write the message to the ROS1 bag
                        writer.write(conn_map[conn.topic], timestamp, rawdata)
                        pbarW.update(1)
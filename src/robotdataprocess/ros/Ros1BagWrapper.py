from ..data_types.Data import Data, ROSMsgLibType
from decimal import Decimal
import glob
import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader as Reader1
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from rosbags.typesys.store import Typestore
from typeguard import typechecked
from typing import Union, List, Dict


class Ros1BagWrapper:
    """
    Wrapper around a ROS1 .bag file, providing typestore management and
    topic message counting. Mirrors the read-side of Ros2BagWrapper.
    """

    # Define attributes
    bag_path: Path
    external_msgs_path: Union[Path, None]
    typestore1: Typestore

    @typechecked
    def __init__(self, bag_path: Union[Path, str], external_msgs_path: Union[Path, str, None]):
        """
        Args:
            bag_path: Path to the ROS1 .bag file.
            external_msgs_path: Path to the directory containing external message definitions.
        """

        if isinstance(bag_path, str): self.bag_path = Path(bag_path)
        else: self.bag_path = bag_path
        if isinstance(external_msgs_path, str): self.external_msgs_path = Path(external_msgs_path)
        else: self.external_msgs_path = external_msgs_path

        # Create ROS1 Typestore
        self.typestore1 = self._create_typestore_with_external_msgs(Stores.ROS1_NOETIC, self.external_msgs_path)

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
            store_type (Stores): The rosbags Store type.
            external_msgs_path: Path to directory to search for additional messages to add.
        Returns:
            Typestore: The typestore with all external messages added.
        """

        def register_msgs(path: Path, typestore: Typestore):
            for msg_path in glob.glob(str(path / Path('**') / Path('*.msg')), recursive=True):
                msg_path = Path(msg_path)
                msgdef = msg_path.read_text(encoding='utf-8')
                typestore.register(get_types_from_msg(msgdef, Ros1BagWrapper._guess_msgtype(msg_path)))

        typestore = get_typestore(store_type)

        try: register_msgs(external_msgs_path, typestore)
        except: print("No `external_msgs_path` path provided, using no external msgs for ros1 Typestore")
        return typestore

    def get_typestore(self) -> Typestore:
        """Return the ROS1 typestore."""
        return self.typestore1

    # =========================================================================
    # =============================== Metadata ================================
    # =========================================================================

    def get_topic_count(self, topic_name: str) -> int:
        """
        Get the number of messages on the specified topic by summing the message
        count across all connections for that topic.

        Args:
            topic_name (str): The name of the topic.
        Returns:
            int: Number of messages.
        Raises:
            ValueError: If topic_name doesn't exist in the bag.
        """

        with Reader1(self.bag_path) as reader:
            count = sum(c.msgcount for c in reader.connections if c.topic == topic_name)

        if count == 0:
            raise ValueError(f"Topic '{topic_name}' doesn't exist in the bag or has 0 messages!")
        return count

    def list_topics(self) -> Dict[str, int]:
        """
        Return a dictionary mapping each topic name to its message count.

        Returns:
            Dict[str, str]: topic name -> message count.
        """

        topics: Dict[str, int] = {}
        with Reader1(self.bag_path) as reader:
            for conn in reader.connections:
                topics[conn.topic] = topics.get(conn.topic, 0) + conn.count
        return topics

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
            raise ValueError(f"Message does not have a valid header with timestamp: {msg}")

from ..data_types.Data import Data
from decimal import Decimal
from multiprocessing import Process
import os
import time
import traceback
from typeguard import typechecked
from typing import List, Union

@typechecked
def _run_ROS2_publisher_process(data: Data, topic_name: str, core_id: Union[int, None] = None):
    """
    Entry point for each ROS2 publishing multiprocessing worker.

    Args:
        data: The Data object to publish.
        topic_name: The ROS2 topic to publish on.
        core_id: Optional CPU core index to pin this process to.
    """

    # CPU isolation first (before ROS/dataset init)
    if core_id is not None:
        try:
            os.sched_setaffinity(0, {core_id})
            print(f"[Worker] CPU affinity set to core {core_id}")
        except Exception as e:
            print(f"[Worker] Failed to set CPU affinity: {e}")

    try:
        # Lazy import rclpy ONLY here
        import rclpy
        from rclpy.node import Node

        class SingleDataPublisher(Node):
            """
            ROS2 node that publishes messages from a single Data object
            honoring timestamps using a high-frequency timer.
            """

            def __init__(self, data: Data, topic_name: str):
                super().__init__(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}")
                self.data = data
                self.topic = topic_name

                # Create publisher
                self.publisher = self.create_publisher(self.data.get_ros_msg_type(), self.topic, 10)

                # Timing setup
                self.index = 0
                self.start_time = Decimal(time.monotonic())
                self.first_ts = Decimal(self.data.timestamps[0])

                # Prebuild the first message
                self.next_msg = self.data.get_ros_msg(0)

                # High-resolution timer (fires every 500 μs)
                self.timer = self.create_timer(0.0005, self.timer_callback)

            def timer_callback(self):
                # Check if we have no more messages to publish
                if self.index >= self.data.len():
                    # Shut down cleanly
                    self.get_logger().info("Finished publishing. Shutting down.")
                    self.timer.cancel()
                    rclpy.shutdown()
                    return

                # Calculate target publish time for the current message
                now = Decimal(time.monotonic())
                target: Decimal = (self.data.timestamps[self.index] - self.first_ts + self.start_time)

                # Publish when time has arrived
                if now >= target:
                    self.publisher.publish(self.next_msg)
                    self.index += 1

                    # Prepare the next message
                    if self.index < self.data.len():
                        self.next_msg = self.data.get_ros_msg(self.index)
                    else:
                        self.next_msg = None

        # Start ROS2 node
        rclpy.init()
        node = SingleDataPublisher(data, topic_name)
        rclpy.spin(node)

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def publish_data_ROS2_multiprocess(data_list: List[Data], topics: List[str]) -> None:
    """
    Launches one ROS2 publisher process per Data stream.

    Args:
        data_list: list of Data objects
        topics: list of ROS topic names
    """

    # Ensure we have matching data and topics
    assert len(data_list) == len(topics)

    # Assign CPU affinities automatically
    cpu_affinities: List[int] = list(range(len(data_list)))

    processes: List[Process] = []
    for data, topic, core in zip(data_list, topics, cpu_affinities):
        p = Process(target=_run_ROS2_publisher_process,
            args=(data, topic, core))
        p.start()
        processes.append(p)

    # Wait for all publishers to finish
    for p in processes:
        p.join()

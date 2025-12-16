from ..data_types.Data import Data, ROSMsgLibType
from ..data_types.ImageData.ImageData import ImageData
from decimal import Decimal
from multiprocessing import Process, Manager
import queue as queue_module
import os
import time
import traceback
from typeguard import typechecked
from typing import List, Union, Any, Tuple

QUEUE_SIZE = 10
TIMER_FREQ = 2000  # Hz
MSG_BUFFER_MAX_VAL = 100

@typechecked
def _run_ROS_publisher_process(data: Data, topic_name: str, print_line_num: int, num_workers: int = 1) -> None:
    """
    Entry point for each ROS1 publishing multiprocessing worker.

    Args:
        data: The Data object to publish.
        topic_name: The ROS topic to publish on.
        print_line_num: The line number in the terminal to print status updates on.
        num_workers: Number of worker processes to pre-build messages.
    """

    try:
        # Lazy import rospy ONLY here
        import rospy

        class SingleDataPublisherROS1():
            def __init__(self, data: Data, topic_name: str, print_line_num: int, num_workers: int = 1):
                # Save parameters
                self.data = data
                self.topic = topic_name
                self.print_line_num = print_line_num
                self.num_workers = num_workers

                # Timing setup
                self.index = 0
                self.start_time = Decimal(time.monotonic())
                self.first_ts = Decimal(self.data.timestamps[0])
                self.prev_time = self.start_time
                self.total_intervals = []

                # Message queue to hold build messages from workers
                self.manager = Manager()
                self.msg_buf = self.manager.dict()
                self.buf_size = self.manager.Value('i', 0)
                self.next_msg = None
                self._last_pub = None

                # Start worker processes to pre-build messages
                self.workers: List[Process] = []
                for worker_id in range(self.num_workers):
                    p = Process(target=self._message_worker, args=(worker_id,))
                    p.start()
                    self.workers.append(p)

                # After creating sub-process (since rospy cannot be safely forked after init), initialize rospy
                rospy.init_node(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}", anonymous=True)

                # Create publisher (TODO: Look into QoS settings)
                self.publisher = rospy.Publisher(self.topic, self.data.get_ros_msg_type(ROSMsgLibType.ROSPY), queue_size=QUEUE_SIZE)

                # High-resolution rate for triggering publishes
                self.rate = rospy.Rate(TIMER_FREQ)
                while not rospy.is_shutdown():
                    self._timer_callback()
                    self.rate.sleep()
                rospy.signal_shutdown("Node shutdown...")

            def _message_worker(self, worker_id: int):
                """
                Worker process: prebuild ROS messages and put them into the queue.
                Each worker handles every nth message starting at worker_id.
                """

                # Initialize this worker's rospy node
                rospy.init_node(f"robotdataprocess_worker_{self.topic.replace('/', '_')}_{worker_id}", anonymous=True)

                # Load messages
                for idx in range(worker_id, self.data.len(), self.num_workers):
                    msg = self.data.get_ros_msg(ROSMsgLibType.ROSPY, idx)
                    timestamp = Decimal(self.data.timestamps[idx])

                    while True:
                        if self.buf_size.value < MSG_BUFFER_MAX_VAL: 
                            self.msg_buf[idx] = (float(timestamp), msg)
                            self.buf_size.value += 1
                            break
                        else:
                            time.sleep(1 / float(TIMER_FREQ))

            def _pop_msg_with_index(self, target_index: int) -> Union[Tuple, None]:
                """
                Search queue for (idx, ts, msg) where idx == target_index.
                Returns the item or None if not found.
                Preserves the ordering of all other items.
                """

                if target_index in self.msg_buf:
                    item = self.msg_buf.pop(target_index)
                    self.buf_size.value -= 1
                    idx = target_index
                    ts, msg = item
                    return (idx, ts, msg)
                return None
        
            def _timer_callback(self):
                # Check if we have no more messages to publish
                if self.index >= self.data.len():
                    # Shut down cleanly
                    rospy.loginfo("Finished publishing. Shutting down...")
                    rospy.signal_shutdown("Node shutdown...")
                    return
                
                # Get the next message from the queue if we don't have one ready
                if self.next_msg is None:
                    result = self._pop_msg_with_index(self.index)
                    if result is None: return
                    self.next_msg: Tuple[int, float, Any] = result

                # Calculate target publish time for the current message
                now = Decimal(time.monotonic())
                target: Decimal = (self.data.timestamps[self.index] - self.first_ts + self.start_time)

                # Publish when time has arrived
                while now >= target:
                    self.publisher.publish(self.next_msg[2])
                    self.index += 1

                    # Check if we've published timestamps out of order
                    if self._last_pub is not None and self._last_pub[1] >= self.next_msg[1]:
                        raise RuntimeError("Published timestamps out of order!")
                    self._last_pub = self.next_msg
                    
                    # Stats calculation
                    elapsed = now - self.start_time
                    msgs_published = self.index + 1
                    deviation = float(now - target)
                    interval = float(now - self.prev_time) if self.index > 0 else 0.0
                    self.prev_time = now
                    self.total_intervals.append(interval)
                    avg_hz = msgs_published / float(elapsed) if elapsed > 0 else 0.0
                    inst_hz = 1.0 / interval if interval > 0 else 0.0

                    # Print single-line summary
                    print(f"\033[{self.print_line_num + 2};0H\033[2K", end='')  # move + clear line
                    print(
                        f"\rTopic: {self.topic} | Published: {msgs_published}/{self.data.len()} | "
                        f"Avg Hz: {avg_hz:.2f} | "
                        f"Inst Hz: {inst_hz:.2f} | Deviation: {deviation*1:.6f} s",
                        end='\n',
                        flush=False
                    )

                    # Prepare the next message
                    if self.index < self.data.len():
                        result = self._pop_msg_with_index(self.index)
                        if result is None: 
                            self.next_msg = None
                            return
                        self.next_msg = result
                    else:
                        self.next_msg = None

                    # Calculate target publish time for the next message
                    if self.index >= self.data.len():
                        break
                    now = Decimal(time.monotonic())
                    target: Decimal = (self.data.timestamps[self.index] - self.first_ts + self.start_time)
        
        publisher = SingleDataPublisherROS1(data, topic_name, print_line_num, num_workers)

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def _run_ROS2_publisher_process(data: Data, topic_name: str, print_line_num: int, num_workers: int = 1) -> None:
    """
    Entry point for each ROS2 publishing multiprocessing worker.

    Args:
        data: The Data object to publish.
        topic_name: The ROS2 topic to publish on.
        print_line_num: The line number in the terminal to print status updates on.
        num_workers: Number of worker processes to pre-build messages.
    """

    try:
        # Lazy import rclpy ONLY here
        import rclpy
        from rclpy.node import Node

        class SingleDataPublisherROS2(Node):
            """
            ROS2 node that publishes messages from a single Data object
            honoring timestamps using a high-frequency timer.
            """

            def __init__(self, data: Data, topic_name: str, print_line_num: int, num_workers: int = 1):
                super().__init__(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}")
                self.data = data
                self.topic = topic_name
                self.print_line_num = print_line_num
                self.num_workers = num_workers

                # Create publisher (TODO: Look into QoS settings)
                self.publisher = self.create_publisher(self.data.get_ros_msg_type(ROSMsgLibType.RCLPY), self.topic, QUEUE_SIZE)

                # Timing setup
                self.index = 0
                self.start_time = Decimal(time.monotonic())
                self.first_ts = Decimal(self.data.timestamps[0])
                self.prev_time = self.start_time
                self.total_intervals = []

                # Message queue to hold build messages from workers
                self.manager = Manager()
                self.msg_buf = self.manager.dict()
                self.buf_size = self.manager.Value('i', 0)
                self.next_msg = None
                self._last_pub = None

                # Start worker processes to pre-build messages
                self.workers: List[Process] = []
                for worker_id in range(self.num_workers):
                    p = Process(target=self._message_worker, args=(worker_id,))
                    p.start()
                    self.workers.append(p)

                # High-resolution timer
                self.timer = self.create_timer(1.0 / float(TIMER_FREQ), self._timer_callback)

            def _message_worker(self, worker_id: int):
                """
                Worker process: prebuild ROS messages and put them into the queue.
                Each worker handles every nth message starting at worker_id.
                """

                for idx in range(worker_id, self.data.len(), self.num_workers):
                    msg = self.data.get_ros_msg(ROSMsgLibType.RCLPY, idx)
                    timestamp = Decimal(self.data.timestamps[idx])

                    while True:
                        if self.buf_size.value < MSG_BUFFER_MAX_VAL: 
                            self.msg_buf[idx] = (float(timestamp), msg)
                            self.buf_size.value += 1
                            break
                        else:
                            time.sleep(1 / float(TIMER_FREQ))

            def _pop_msg_with_index(self, target_index: int) -> Union[Tuple, None]:
                """
                Search queue for (idx, ts, msg) where idx == target_index.
                Returns the item or None if not found.
                Preserves the ordering of all other items.
                """

                if target_index in self.msg_buf:
                    item = self.msg_buf.pop(target_index)
                    self.buf_size.value -= 1
                    idx = target_index
                    ts, msg = item
                    return (idx, ts, msg)
                return None
            
            def _timer_callback(self):
                # Check if we have no more messages to publish
                if self.index >= self.data.len():
                    # Shut down cleanly
                    self.get_logger().info("Finished publishing. Shutting down.")
                    self.timer.cancel()
                    rclpy.shutdown()
                    return
                
                # Get the next message from the queue if we don't have one ready
                if self.next_msg is None:
                    result = self._pop_msg_with_index(self.index)
                    if result is None: return
                    self.next_msg: Tuple[int, float, Any] = result

                # Calculate target publish time for the current message
                now = Decimal(time.monotonic())
                target: Decimal = (self.data.timestamps[self.index] - self.first_ts + self.start_time)

                # Publish when time has arrived
                while now >= target:
                    self.publisher.publish(self.next_msg[2])
                    self.index += 1

                    # Check if we've published timestamps out of order
                    if self._last_pub is not None and self._last_pub[1] >= self.next_msg[1]:
                        raise RuntimeError("Published timestamps out of order!")
                    self._last_pub = self.next_msg
                    
                    # Stats calculation
                    elapsed = now - self.start_time
                    msgs_published = self.index + 1
                    deviation = float(now - target)
                    interval = float(now - self.prev_time) if self.index > 0 else 0.0
                    self.prev_time = now
                    self.total_intervals.append(interval)
                    avg_hz = msgs_published / float(elapsed) if elapsed > 0 else 0.0
                    inst_hz = 1.0 / interval if interval > 0 else 0.0

                    # Print single-line summary
                    print(f"\033[{self.print_line_num + 2};0H\033[2K", end='')  # move + clear line
                    print(
                        f"\rTopic: {self.topic} | Published: {msgs_published}/{self.data.len()} | "
                        f"Avg Hz: {avg_hz:.2f} | "
                        f"Inst Hz: {inst_hz:.2f} | Deviation: {deviation*1:.6f} s",
                        end='\n',
                        flush=False
                    )

                    # Prepare the next message
                    if self.index < self.data.len():
                        result = self._pop_msg_with_index(self.index)
                        if result is None: 
                            self.next_msg = None
                            return
                        self.next_msg = result
                    else:
                        self.next_msg = None

                    # Calculate target publish time for the next message
                    if self.index >= self.data.len():
                        break
                    now = Decimal(time.monotonic())
                    target: Decimal = (self.data.timestamps[self.index] - self.first_ts + self.start_time)

        # Start ROS2 node
        rclpy.init()
        node = SingleDataPublisherROS2(data, topic_name, print_line_num, num_workers)
        rclpy.spin(node)

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def publish_data_ROS_multiprocess(data_list: List[Data], topics: List[str], libtype: ROSMsgLibType) -> None:
    """
    Launches one publisher process per Data stream, either for ROS1 or ROS2.

    Args:
        data_list: list of Data objects
        topics: list of ROS topic names
        libtype: ROSMsgLibType indicating whether to use rospy (ROS1) or rclpy (ROS2).
    """

    # Ensure we have matching data and topics
    assert len(data_list) == len(topics)

    # Clear the screen for status output
    print("\033[2J") 

    # Assign line nubmers for each publisher to print status updates
    print_line_nums: List[int] = list(range(len(data_list)))

    # Launch the appropriate publisher processes
    processes: List[Process] = []
    for data, topic, print_line_num in zip(data_list, topics, print_line_nums):
        if isinstance(data, ImageData): num_workers = 2
        else: num_workers = 1

        if libtype == ROSMsgLibType.RCLPY:
            pub_proc_func = _run_ROS2_publisher_process
        elif libtype == ROSMsgLibType.ROSPY:
            pub_proc_func = _run_ROS_publisher_process

        p = Process(target=pub_proc_func, args=(data, topic, print_line_num, num_workers))
        p.start()
        processes.append(p)

    # Wait for all publishers to finish
    for p in processes:
        p.join()

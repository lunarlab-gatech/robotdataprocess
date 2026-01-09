from ..data_types.Data import Data, ROSMsgLibType
from ..data_types.ImageData.ImageData import ImageData
from decimal import Decimal
from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy
import queue as queue_module
import os
from rich.live import Live
from rich.table import Table
from rich.console import Console
import time
import traceback
from typeguard import typechecked
from typing import List, Union, Any, Tuple

QUEUE_SIZE = 10
TIMER_FREQ = 2000 # Hz
MSG_BUFFER_MAX_VAL = 100 # entries
RESTART_JUMP_MSGS = 5 # msgs

class _SingleDataPublisher():
    """ 
    Single Data Publisher that is used for both ROS1 and ROS2 implementations.
    Publishes messages from a single Data object honoring timestamps using a high-frequency timer.

    Args:
        libtype: ROSMsgLibType indicating whether to use rospy (ROS1) or rclpy (ROS2).
        ros2_node_class: If ROS2, this is the Node class.
        data: The Data object to publish.
        topic_name: The ROS topic to publish on.
        type: The ROS message type for data that can be published as multiple types.
        num_workers: Number of worker processes to pre-build messages.
        verbose: Whether to print topic publishing status to the console.
        stats_dict: Dictionary for processes to put publishing statistics for printing.
    """

    def __init__(self, libtype: ROSMsgLibType, ros2_node_class: Union[Any, None], data: Data, topic_name: str, 
                 type: Union[str, None], num_workers: int = 1, verbose: bool = True, stats_dict: Union[DictProxy, None] = None):
        
        # Save parameters
        self.libtype = libtype
        self.ros2_node_class = ros2_node_class
        self.data = data
        self.topic = topic_name
        self.type = type
        self.num_workers = num_workers
        self.verbose = verbose
        self.stats_dict = stats_dict
        self.skipped_msgs = 0
        self._is_finished = False
        self.restart_when_behind = False

        # Timing setup
        self.start_time = Decimal(time.monotonic()) + Decimal('1.0')
        self.first_ts = Decimal(self.data.timestamps[0])
        self.prev_time = self.start_time
        self.total_intervals = []

        # Message queue to hold build messages from workers
        self.manager = Manager()
        self.msg_buf = self.manager.dict()
        self.buf_size = self.manager.Value('i', 0)
        self.index = self.manager.Value("i", 0)
        self.next_msg = None
        self._last_pub = None

        # Start worker processes to pre-build messages
        self.workers: List[Process] = []
        for worker_id in range(self.num_workers):
            p = Process(target=self._message_worker, args=(worker_id,))
            p.start()
            self.workers.append(p)

        # Get data type
        if self.type is not None:
            self._pub_type = self.data.get_ros_msg_type(self.libtype, self.type)
        else:
            self._pub_type = self.data.get_ros_msg_type(self.libtype)

        # Create publisher (TODO: Look into QoS settings)
        if self.libtype == ROSMsgLibType.ROSPY:
            # For ROS1, we have to intialize rospy AFTER forking processes
            import rospy
            rospy.init_node(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}", anonymous=True)
            self.publisher = rospy.Publisher(self.topic, self._pub_type, queue_size=QUEUE_SIZE)

        elif self.libtype == ROSMsgLibType.RCLPY:
            self.publisher = self.ros2_node_class.create_publisher(self._pub_type, self.topic, QUEUE_SIZE)

        # High-resolution timer for triggering publishes
        if self.libtype == ROSMsgLibType.ROSPY:
            import rospy
            self.rate = rospy.Rate(TIMER_FREQ)
            while not rospy.is_shutdown() and not self._is_finished:
                self._timer_callback()
                self.rate.sleep()

        elif self.libtype == ROSMsgLibType.RCLPY:
            self.timer = self.ros2_node_class.create_timer(1.0 / float(TIMER_FREQ), self._timer_callback)

    def _message_worker(self, worker_id: int):
        """
        Worker process: prebuild ROS messages and put them into the queue.
        Each worker handles every nth message starting at worker_id.
        """

        # For ROS1, initialize this worker's rospy node
        if self.libtype == ROSMsgLibType.ROSPY:
            import rospy
            rospy.init_node(f"robotdataprocess_worker_{self.topic.replace('/', '_')}_{worker_id}", anonymous=True)

        # Iterature though entries assigned to this worker
        for idx in range(worker_id, self.data.len(), self.num_workers):
            # Skip messages that the main thread has skipped
            if idx < self.index.value:
                continue
            
            # Load the message
            if self.type is not None:
                msg = self.data.get_ros_msg(self.libtype, idx, self.type)
            else:
                msg = self.data.get_ros_msg(self.libtype, idx)
            timestamp = Decimal(self.data.timestamps[idx])

            # Put the message into the buffer
            while True:
                if self.buf_size.value < MSG_BUFFER_MAX_VAL: 
                    self.msg_buf[idx] = (float(timestamp), msg)
                    self.buf_size.value += 1
                    break
                elif idx < self.index.value:
                    break
                else:
                    time.sleep(1 / float(TIMER_FREQ))
    
    def _pop_msg_with_index(self, target_index: int) -> Union[Tuple, None]:
        """
        Search queue for (idx, ts, msg) where idx == target_index.
        Returns the item or None if not found. Additionally clears stale messages.
        """

        # Clear out any stale messages that were skipped
        keys_to_delete = [k for k in self.msg_buf.keys() if k < target_index]
        for k in keys_to_delete:
            self.msg_buf.pop(k)
            self.buf_size.value -= 1

        # Get the actual target we're looking for
        if target_index in self.msg_buf:
            item = self.msg_buf.pop(target_index)
            self.buf_size.value -= 1
            idx = target_index
            ts, msg = item
            return (idx, ts, msg)
        return None
    
    def _timer_callback(self):
        # Check if we have no more messages to publish
        if self.index.value >= self.data.len():
            # Shut down cleanly
            if self.libtype == ROSMsgLibType.ROSPY:
                import rospy
                rospy.loginfo("Finished publishing.")
                rospy.loginfo("Waiting for worker processes to finish...")
                for w in self.workers:
                    w.join()
                rospy.signal_shutdown("Node shutdown...")
                self._is_finished = True
                rospy.loginfo("Node destroyed.")
                return
            elif self.libtype == ROSMsgLibType.RCLPY:
                self.ros2_node_class.get_logger().info("Finished publishing.")
                self.timer.cancel()
                self.ros2_node_class.get_logger().info("Waiting for worker processes to finish...")
                for w in self.workers:
                    w.join()
                self.ros2_node_class.destroy_node()
                self._is_finished = True
                self.ros2_node_class.get_logger().info("Node destroyed.")
                return
        
        # Get the next message from the queue if we don't have one ready
        if self.next_msg is None:
            result = self._pop_msg_with_index(self.index.value)
            if result is None: return
            self.next_msg: Tuple[int, float, Any] = result

        # Calculate target publish time for the current message
        now = Decimal(time.monotonic())
        target: Decimal = (self.data.timestamps[self.index.value] - self.first_ts + self.start_time)

        # Publish when time has arrived
        if now >= target:

            # Check if we are behind
            behind: bool = False
            if self.index.value < self.data.len() - 1:
                next_target = (self.data.timestamps[self.index.value + 1] - self.first_ts + self.start_time)
                if now > next_target:
                    behind = True
            
            # If behind...
            if behind:
                # We either plan a restart
                if self.restart_when_behind:

                    # Calculate the next message to publish (skipping some)
                    prev_skipped_msgs = self.skipped_msgs
                    while self.index.value < self.data.len() - 1 and self.skipped_msgs - prev_skipped_msgs < RESTART_JUMP_MSGS:
                        self.index.value += 1
                        self.skipped_msgs += 1

                    # We don't want to publish before its time, so empty next_msg and return
                    self.next_msg = None
                    return
                
                # Or just skip to the next message that isn't behind
                else:
                    # Find the next index
                    while self.index.value < self.data.len() - 1: 
                        next_target = (self.data.timestamps[self.index.value + 1] - self.first_ts + self.start_time) 
                        if now < next_target: 
                            break 
                        self.index.value += 1 
                        self.skipped_msgs += 1

                    # Load the next message for it (if available)
                    result = self._pop_msg_with_index(self.index.value)
                    if result is None: return
                    self.next_msg: Tuple[int, float, Any] = result

            # Publish the message
            self.publisher.publish(self.next_msg[2])
            self.index.value += 1

            # Check if we've published timestamps out of order
            if self._last_pub is not None and self._last_pub[1] >= self.next_msg[1]:
                raise RuntimeError("Published timestamps out of order!")
            self._last_pub = self.next_msg
            
            # Stats calculation
            elapsed = now - self.start_time
            msgs_published = self.index.value - self.skipped_msgs
            deviation = float(now - target)
            interval = float(now - self.prev_time) if self.index.value > 0 else 0.0
            self.total_intervals.append(interval)
            avg_hz = msgs_published / float(elapsed) if elapsed > 0 else 0.0
            inst_hz = 1.0 / interval if interval > 0 else 0.0
            self.prev_time = now

            # Update statistics
            if self.verbose and self.stats_dict is not None:
                self.stats_dict[self.topic] = {
                    "last_update_time": now,
                    "progress": self.index.value,
                    "total": self.data.len(),
                    "avg_hz": avg_hz,
                    "inst_hz": inst_hz,
                    "deviation": deviation,
                    "prev_time": self.prev_time,
                    "interval": interval,
                    "skipped": self.skipped_msgs
                }

            # Prepare the next message
            if self.index.value < self.data.len():
                result = self._pop_msg_with_index(self.index.value)
                if result is None: 
                    self.next_msg = None
                    return
                self.next_msg = result
            else:
                self.next_msg = None

@typechecked
def _run_ROS_publisher_process(data: Data, topic_name: str, type: Union[str, None], num_workers: int = 1, 
                               verbose: bool = True, stats_dict: Union[DictProxy, None] = None) -> None:
    """
    Entry point for each ROS1 publishing multiprocessing worker. 
    NOTE: This function feels useless, but follows similar structure to the ROS2 version, which is more complex.
    """

    try:
        class SingleDataPublisherROS1():
            def __init__(self, data: Data, topic_name: str, type: Union[str, None], num_workers: int = 1, 
                         verbose: bool = True, stats_dict: Union[DictProxy, None] = None):
                self._pub = _SingleDataPublisher(ROSMsgLibType.ROSPY, None, data, topic_name, type, num_workers, verbose, stats_dict)
        publisher = SingleDataPublisherROS1(data, topic_name, type, num_workers, verbose, stats_dict)
    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def _run_ROS2_publisher_process(data: Data, topic_name: str, type: Union[str, None], num_workers: int = 1, 
                                verbose: bool = True, stats_dict: Union[DictProxy, None] = None) -> None:
    """
    Entry point for each ROS2 publishing multiprocessing worker.
    """

    try:
        # Lazy import rclpy ONLY here
        import rclpy
        from rclpy.node import Node

        # Wrapper class that is also a ROS2 Node, so that we are in compliance with rclpy design.
        class SingleDataPublisherROS2(Node):
            def __init__(self, data: Data, topic_name: str, type: Union[str, None], num_workers: int = 1, 
                         verbose: bool = True, stats_dict: Union[DictProxy, None] = None):
                super().__init__(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}")
                self._pub = _SingleDataPublisher(ROSMsgLibType.RCLPY, self, data, topic_name, type, num_workers, verbose, stats_dict)

            def is_finished(self) -> bool:
                return self._pub._is_finished
     
        # Start ROS2 node
        if not rclpy.ok():
            rclpy.init()
        node = SingleDataPublisherROS2(data, topic_name, type, num_workers, verbose, stats_dict)
        print("Spinning ROS2 node...")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=2.0)
            if node.is_finished():
                break
        print("ROS2 node spin complete.")

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def publish_data_ROS_multiprocess(data_list: List[Data], data_topics: List[str], data_msg_type: List[Union[str, None]], 
                                  data_workers: List[int], libtype: ROSMsgLibType, shutdown_ros: bool, 
                                  verbose: bool = True, delay_seconds: float = 0.0) -> None:
    """
    Launches one publisher process per Data stream, either for ROS1 or ROS2.

    Args:
        data_list: list of Data objects
        data_topics: list of ROS topic names
        data_msg_type: list of ROS message types for data that can be published as multiple types.
        data_workers: Each topic will have one publisher process and X number of worker processes to pull data.
        libtype: ROSMsgLibType indicating whether to use rospy (ROS1) or rclpy (ROS2).
        shutdown_ros: Whether to shutdown ROS after publishing is complete.
        verbose: Whether to print topic publishing status to the console.
        delay_seconds: Delay in seconds before doing anything (for testing purposes).
    """

    # Optional delay before starting (for testing purposes)
    if delay_seconds > 0.0:
        time.sleep(delay_seconds)

    # Ensure we have matching data and topics
    assert len(data_list) == len(data_topics)

    # Create a shared dictionary across processes for statistics
    manager = Manager()
    stats_dict = manager.dict() # Shared across processes

    # Initialize stats_dict for each topic
    for topic in data_topics:
        stats_dict[topic] = {"last_update_time": 0, "progress": 0, "total": 0, "avg_hz": 0, 
                             "inst_hz": 0, "deviation": 0,
                             "prev_time": 0, "interval": 0, "skipped": 0}

    # Launch the appropriate publisher processes
    processes: List[Process] = []
    topic_to_proc: dict = {}
    for data, topic, type, workers in zip(data_list, data_topics, data_msg_type, data_workers):
        if libtype == ROSMsgLibType.RCLPY:
            pub_proc_func = _run_ROS2_publisher_process
        elif libtype == ROSMsgLibType.ROSPY:
            pub_proc_func = _run_ROS_publisher_process

        p = Process(target=pub_proc_func, args=(data, topic, type, workers, verbose, stats_dict))
        p.start()
        processes.append(p)
        topic_to_proc[topic] = p

    # Helper to build the table with a Status column
    def generate_table() -> Table:
        table = Table(title="ROS Publisher Dashboard", show_header=True, header_style="bold magenta")
        table.add_column("Status", justify="center", min_width=14)
        table.add_column("Topic", style="cyan")
        table.add_column("Progress", justify="right")
        table.add_column("Avg Hz", justify="right")
        table.add_column("Inst Hz", justify="right")
        table.add_column("Skipped Msgs", justify="right")
        
        for topic, s in stats_dict.items():
            proc = topic_to_proc[topic]
            row_style = None

            if proc.is_alive():
                if time.monotonic() - float(s["last_update_time"]) < 0.1:
                    status = "[bold green]RUNNING[/bold green]"
                else:
                    status = "[bold yellow]UNRESPONSIVE[/bold yellow]"
                    row_style = "grey50"
            elif s['progress'] >= s['total'] and s['total'] > 0:
                status = "[bold blue]FINISHED[/bold blue]"
            else:
                status = "[bold red]STOPPED[/bold red]"

            table.add_row(
                status,
                topic, 
                f"{s['progress']}/{s['total']}", 
                f"{s['avg_hz']:.1f}", 
                f"{s['inst_hz']:.1f}",
                f"{s['skipped']}",
                style=row_style
            )
        return table

    # Wait for all publishers to finish (and provide updates if requested)
    if verbose:
        with Live(generate_table(), refresh_per_second=10, screen=False) as live:
            while any(p.is_alive() for p in processes):
                live.update(generate_table())
                time.sleep(0.1)
            live.update(generate_table())
        for p in processes:
            p.join()
    else:
        for p in processes:
            p.join()
    
    # Shutdown if requested
    if libtype == ROSMsgLibType.RCLPY and shutdown_ros:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
            print("✓ ROS2 shutdown complete.")
    elif libtype == ROSMsgLibType.ROSPY and shutdown_ros:
        # ROS1 shutdown handled in each publisher node
        pass
    elif shutdown_ros:
        raise NotImplementedError(f"Unsupported ROSMsgLibType {libtype} for shutdown_ros parameter!")
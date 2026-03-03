from ..data_types.Data import ROSMsgLibType
from ..data_types.SequentialData import SequentialData
from ..data_types.ImageData.ImageData import ImageData
from ..ModuleImporter import ModuleImporter
import multiprocessing
from multiprocessing import Process, Manager, Event, resource_tracker
from multiprocessing.managers import DictProxy
import multiprocessing.shared_memory
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import queue as queue_module
import os
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.markup import escape
import signal
import time
import traceback
from typeguard import typechecked
from typing import List, Union, Any, Tuple

ROS_PUB_QUEUE_SIZE = 10
MSG_QUEUE_MAX_SIZE = 20 # entries
RESTART_JUMP_MSGS = 5 # msgs
CLOCK_TOPIC = '/clock'
CLOCK_HZ = 100.0

def neutralize_resource_tracker():
    # Create a dummy tracker class that won't crash
    class DummyTracker:
        def register(self, *args, **kwargs): pass
        def unregister(self, *args, **kwargs): pass
        def ensure_running(self, *args, **kwargs): pass

    # 2. Globally neutralize the actual resource_tracker
    resource_tracker.register = lambda *args, **kwargs: None
    resource_tracker.unregister = lambda *args, **kwargs: None

    # 3. Specifically for Python 3.10+, replace the module reference 
    # instead of setting it to None
    multiprocessing.shared_memory.resource_tracker = DummyTracker()

neutralize_resource_tracker()

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
        hertz: The expected frequency to publish at (used to set the publisher timer frequency).
        stop_event: Allow sub-processes to be stopped by KeyboardInterrupts.
        wait_for_sub: Whether to wait for a subscriber until we start publishing (used for tests).
        num_workers: Number of worker processes to pre-build messages.
        verbose: Whether to print topic publishing status to the console.
        stats_dict: Dictionary for processes to put publishing statistics for printing.
    """

    def __init__(self, libtype: ROSMsgLibType, ros2_node_class: Union[Any, None], data: SequentialData, topic_name: str,
                 type: Union[str, None], hertz: float, stop_event, wait_for_sub: bool = False, num_workers: int = 1,
                 verbose: bool = True, stats_dict: Union[DictProxy, None] = None, latched: bool = False):
        
        # Save parameters
        self.libtype = libtype
        self.ros2_node_class = ros2_node_class
        self.data = data
        self.topic = topic_name
        self.type = type
        self.hertz = hertz
        self.timer_freq = 2 * self.hertz
        self.stop_event = stop_event
        self.num_workers = num_workers
        self.verbose = verbose
        self.stats_dict = stats_dict
        self.skipped_msgs = 0
        self._is_finished = False
        self.restart_when_behind = False
        self.latched = latched

        # Timing setup
        self.start_time = time.monotonic() + 1.0
        self.first_ts = float(self.data.timestamps[0])
        self.prev_time = self.start_time
        self.prev_console_update_time = self.start_time

        # Message queue to hold build messages from workers
        self.worker_queue = multiprocessing.Queue(maxsize=MSG_QUEUE_MAX_SIZE)
        self.local_buf: dict = {}
        self.manager = Manager()
        self.index = self.manager.Value("i", 0)
        self.next_msg = None
        self._last_pub = None

        # Determine whether to use shared memory
        if self.libtype == ROSMsgLibType.ROSPY:
            use_shared_mem = True # Leads to large speedups for ROS1
        elif self.libtype == ROSMsgLibType.RCLPY:
            use_shared_mem = False # Empirically, leads to large slowdowns for ROS2

        # Start worker processes to pre-build messages
        self.stop_event_workers = Event()
        self.workers: List[Process] = []
        for worker_id in range(self.num_workers):
            p = Process(target=self._message_worker, args=(worker_id, self.stop_event_workers, self.stats_dict, use_shared_mem))
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
            self.publisher = rospy.Publisher(self.topic, self._pub_type, queue_size=ROS_PUB_QUEUE_SIZE, latch=latched)

        elif self.libtype == ROSMsgLibType.RCLPY:
            if latched:
                from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
                qos = QoSProfile(depth=ROS_PUB_QUEUE_SIZE,
                                 durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=QoSReliabilityPolicy.RELIABLE)
                self.publisher = self.ros2_node_class.create_publisher(self._pub_type, self.topic, qos)
            else:
                self.publisher = self.ros2_node_class.create_publisher(self._pub_type, self.topic, ROS_PUB_QUEUE_SIZE)

        # If desired, wait until we have at least one subscriber to start publishing
        if wait_for_sub:
            if self.libtype == ROSMsgLibType.ROSPY:
                rospy = ModuleImporter.get_module('rospy')
                while self.publisher.get_num_connections() == 0 and not rospy.is_shutdown():
                    rospy.sleep(1.0 / self.timer_freq)
            elif self.libtype == ROSMsgLibType.RCLPY:
                while self.publisher.get_subscription_count() == 0:
                    rclpy = ModuleImporter.get_module('rclpy')
                    rclpy.spin_once(self.ros2_node_class, timeout_sec=1.0 / self.timer_freq)

            # Redo timing setup once we have our subscriber
            self.start_time = time.monotonic() + 1.0
            self.prev_time = self.start_time
            self.prev_console_update_time = self.start_time

        # High-resolution timer for triggering publishes
        if self.libtype == ROSMsgLibType.ROSPY:
            rospy = ModuleImporter.get_module('rospy')
            rospy.Timer(rospy.Duration(1.0 / float(self.timer_freq)), self._timer_callback)
        elif self.libtype == ROSMsgLibType.RCLPY:
            self.timer = self.ros2_node_class.create_timer(1.0 / float(self.timer_freq), self._timer_callback)

        self.log_msg_to_ros(f"robotdataprocess_publisher_{topic_name.replace('/', '_')} intialized!", stats_dict)

    def log_msg_to_ros(self, msg: str, stats_dict: Union[DictProxy, None] = None):
        """
        Log a message to ROS regardless of ROS version (1 or 2).

        Args:
            msg: The message string to log.
            stats_dict: Optional shared manager dict for console logging.
        """

        if self.libtype == ROSMsgLibType.ROSPY:
            rospy = ModuleImporter.get_module('rospy')
            rospy.loginfo(msg)
        elif self.libtype == ROSMsgLibType.RCLPY:
            self.ros2_node_class.get_logger().info(msg)

        # Additionally logs it to the console if enabled
        if stats_dict:
            topic_info = self.stats_dict[self.topic]
            topic_info['log_queue'].append(msg)
            self.stats_dict[self.topic] = topic_info

    def _message_worker(self, worker_id: int, stop_event, stats_dict: Union[DictProxy, None] = None, use_shared_mem: bool = True):
        """
        Worker process: prebuild ROS messages and put them into the queue.
        Each worker handles every nth message starting at worker_id.
        """
        neutralize_resource_tracker()

        # For ROS1, initialize this worker's rospy node
        if self.libtype == ROSMsgLibType.ROSPY:
            import rospy
            rospy.init_node(f"robotdataprocess_worker_{self.topic.replace('/', '_')}_{worker_id}", anonymous=True)

        try:
            # Iterature though entries assigned to this worker
            for idx in range(worker_id, self.data.len(), self.num_workers):
                # Stop working if we recieve a shutdown signal
                if stop_event.is_set():
                    break

                # Skip messages that the main thread has skipped
                if idx < self.index.value:
                    continue
                
                # Load the message
                if self.type is not None:
                    msg = self.data.get_ros_msg(self.libtype, idx, self.type)
                else:
                    msg = self.data.get_ros_msg(self.libtype, idx)
                timestamp = float(self.data.timestamps[idx])

                # If message has data field, use shared memory to transfer to publishing process efficiently
                shm_name = None
                if use_shared_mem and hasattr(msg, 'data') and len(msg.data) > 0:

                    # Write a handler to disable SharedMemory usage if we run out
                    def handler(signum, frame):
                        nonlocal use_shared_mem
                        use_shared_mem = False
                        raise TimeoutError("Calls to SharedMemory freezing! Switching to Process-to-Process Serialization, which will run slower. If in a Docker container, this can be avoided by increasing the shared memory (docker run --shm-size=2gb).")

                    # Set an alarm for 0.2 seconds
                    signal.signal(signal.SIGALRM, handler)
                    signal.setitimer(signal.ITIMER_REAL, 0.2)

                    # Attempt to allocate shared memory
                    if isinstance(msg.data, list): data_to_copy = bytes(msg.data)
                    else: data_to_copy = msg.data
                    pixel_bytes = np.frombuffer(data_to_copy, dtype=np.uint8)
                    success = False
                    try:
                        shm = SharedMemory(create=True, size=pixel_bytes.nbytes)
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
                        success = True
                    except TimeoutError as e:
                        self.log_msg_to_ros(f"{e}", stats_dict)
                    
                    # If successful, save the data to shared memory
                    if success:
                        shared_array = np.ndarray(pixel_bytes.shape, dtype=pixel_bytes.dtype, buffer=shm.buf)
                        shared_array[:] = pixel_bytes[:]
                        
                        # Strip the data from transport
                        msg.data = [] if isinstance(msg.data, list) else b""
                        shm_name = shm.name
                        shm.close()

                # Put the message into the buffer
                while True:
                    if stop_event.is_set() or idx < self.index.value:
                        if shm_name:
                            temp_shm = SharedMemory(name=shm_name)
                            temp_shm.close()
                            temp_shm.unlink()
                        break
                    else:
                        try:
                            payload = (idx, float(timestamp), msg, shm_name)
                            self.worker_queue.put(payload, timeout=1.0 / float(self.timer_freq))
                            break
                        except queue_module.Full:
                            time.sleep(1.0 / float(self.timer_freq))

        except BrokenPipeError:
            # Manager must have died from KeyboardInterrupt
            pass
        self.log_msg_to_ros(f"Worker process {self.topic}_{worker_id} shutting down...", stats_dict)
    
    def _pop_msg_with_index(self, target_index: int) -> Union[Tuple, None]:
        """
        Search queue for (idx, ts, msg) where idx == target_index.
        Returns the item or None if not found. Additionally clears stale messages.
        """

        neutralize_resource_tracker()

        # Drain the Queue into the local reorder buffer if we don't have the msg we want
        if not target_index in self.local_buf:
            while not self.worker_queue.empty():
                try:
                    idx, ts, msg, shm_name = self.worker_queue.get(timeout=1.0 / float(self.timer_freq))
                    if idx < target_index:
                        if shm_name:
                            try:
                                temp_shm = SharedMemory(name=shm_name)
                                temp_shm.close()
                                temp_shm.unlink()
                            except (FileNotFoundError, BufferError):
                                pass
                    else:
                        self.local_buf[idx] = (ts, msg, shm_name)
                except queue_module.Empty:
                    break
        
            # Clear out any stale messages that were skipped
            keys_to_delete = [k for k in self.local_buf.keys() if k < target_index]
            for k in keys_to_delete:
                _, _, stale_shm_name = self.local_buf.pop(k)
                if stale_shm_name:
                    try:
                        temp_shm = SharedMemory(name=stale_shm_name)
                        temp_shm.close()
                        temp_shm.unlink()
                    except (FileNotFoundError, BufferError): 
                        pass

        # Get the actual target we're looking for
        if target_index in self.local_buf:
            ts, msg, shm_name = self.local_buf.pop(target_index)
            
            # Load data from shared memory if shm_name exists
            if shm_name:
                try:
                    actual_shm = SharedMemory(name=shm_name)
                    msg.data = actual_shm.buf.tobytes() 
                    actual_shm.close()
                    actual_shm.unlink()
                except FileNotFoundError:
                    self.log_msg_to_ros(f"Warning: SHM {shm_name} vanished before pop.", self.stats_dict)
                except Exception as e:
                    self.log_msg_to_ros(f"Failed to reconstruct SHM: {e}", self.stats_dict)
                    return None
            
            return (target_index, ts, msg)
        
        # Otherwise, we don't have the target yet...
        return None

    def _timer_callback(self, event=None):
        # Check if we have no more messages to publish or we get a StopEvent:
        if self.index.value >= self.data.len() or self.stop_event.is_set():

            # Latched publishers stay alive until stop_event is set so that
            # late-joining subscribers can still receive the latched messages
            if self.latched and not self.stop_event.is_set():
                return

            # Tell workers to finish (if they haven't already)
            self.stop_event_workers.set()

            # Shut down cleanly
            if self.libtype == ROSMsgLibType.ROSPY:
                import rospy
                self.log_msg_to_ros("Finished publishing.", self.stats_dict)
                self.log_msg_to_ros("Waiting for worker processes to finish...", self.stats_dict)
                for w in self.workers:
                    w.join()
                rospy.signal_shutdown("Node shutdown...")
                self._is_finished = True
                self.log_msg_to_ros("Node destroyed.", self.stats_dict)
                return
            elif self.libtype == ROSMsgLibType.RCLPY:
                self.log_msg_to_ros("Finished publishing.", self.stats_dict)
                self.timer.cancel()
                self.log_msg_to_ros("Waiting for worker processes to finish...", self.stats_dict)
                for w in self.workers:
                    w.join()
                self.ros2_node_class.destroy_node()
                self._is_finished = True
                self.log_msg_to_ros("Node destroyed.", self.stats_dict)
                return
        
        # Get the next message from the queue if we don't have one ready
        if self.next_msg is None:
            result = self._pop_msg_with_index(self.index.value)
            if result is None: return
            self.next_msg: Tuple[int, float, Any] = result

        # Calculate target publish time for the current message
        now = time.monotonic()
        target: float = float(self.data.timestamps[self.index.value]) - self.first_ts + self.start_time

        # Publish when time has arrived
        if now >= target:

            # Check if we are behind
            behind: bool = False
            if self.index.value < self.data.len() - 1:
                next_target = float(self.data.timestamps[self.index.value + 1]) - self.first_ts + self.start_time
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
                        next_target = float(self.data.timestamps[self.index.value + 1]) - self.first_ts + self.start_time
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
            interval = float(now - self.prev_time) if self.index.value > 0 else 0.0
            avg_hz = msgs_published / float(elapsed) if elapsed > 0 else 0.0
            inst_hz = 1.0 / interval if interval > 0 else 0.0
            self.prev_time = now

            # Update statistics (at most 10Hz)
            elapsed_console_update = now - self.prev_console_update_time
            if self.verbose and self.stats_dict is not None and elapsed_console_update > 0.1:
                self.stats_dict[self.topic] = {
                    "last_update_time": now,
                    "progress": self.index.value,
                    "total": self.data.len(),
                    "avg_hz": avg_hz,
                    "inst_hz": inst_hz,
                    "skipped": self.skipped_msgs,
                    "log_queue": self.stats_dict[self.topic]['log_queue'],
                    'local_buf_size': len(self.local_buf),
                    'worker_queue_size': self.worker_queue.qsize()
                }
                self.prev_console_update_time = now

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
def _run_ROS_publisher_process(data: SequentialData, topic_name: str, type: Union[str, None], hertz: float, stop_event, wait_for_sub: bool = False,
                               num_workers: int = 1, verbose: bool = True, stats_dict: Union[DictProxy, None] = None, latched: bool = False) -> None:
    """
    Entry point for each ROS1 publishing multiprocessing worker.
    NOTE: This function feels useless, but follows similar structure to the ROS2 version, which is more complex.
    """

    try:
        import rospy
        class SingleDataPublisherROS1():
            def __init__(self, data: SequentialData, topic_name: str, type: Union[str, None], hertz: float, stop_event, wait_for_sub: bool = False, num_workers: int = 1,
                         verbose: bool = True, stats_dict: Union[DictProxy, None] = None, latched: bool = False):
                self._pub = _SingleDataPublisher(ROSMsgLibType.ROSPY, None, data, topic_name, type, hertz, stop_event, wait_for_sub, num_workers, verbose, stats_dict, latched)
        node = SingleDataPublisherROS1(data, topic_name, type, hertz, stop_event, wait_for_sub, num_workers, verbose, stats_dict, latched)
        while not rospy.is_shutdown() and not node._pub._is_finished:
            time.sleep(0.1)

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def _run_ROS2_publisher_process(data: SequentialData, topic_name: str, type: Union[str, None], hertz: float, stop_event, wait_for_sub: bool = False,
                                num_workers: int = 1, verbose: bool = True, stats_dict: Union[DictProxy, None] = None, latched: bool = False) -> None:
    """
    Entry point for each ROS2 publishing multiprocessing worker.
    """

    try:
        # Lazy import rclpy ONLY here
        import rclpy
        from rclpy.node import Node

        # Wrapper class that is also a ROS2 Node, so that we are in compliance with rclpy design.
        class SingleDataPublisherROS2(Node):
            def __init__(self, data: SequentialData, topic_name: str, type: Union[str, None], hertz: float, stop_event, wait_for_sub: bool = False, num_workers: int = 1,
                         verbose: bool = True, stats_dict: Union[DictProxy, None] = None, latched: bool = False):
                super().__init__(f"robotdataprocess_publisher_{topic_name.replace('/', '_')}")
                self._pub = _SingleDataPublisher(ROSMsgLibType.RCLPY, self, data, topic_name, type, hertz, stop_event, wait_for_sub, num_workers, verbose, stats_dict, latched)

            def is_finished(self) -> bool:
                return self._pub._is_finished

        # Start ROS2 node
        if not rclpy.ok():
            rclpy.init()
        node = SingleDataPublisherROS2(data, topic_name, type, hertz, stop_event, wait_for_sub, num_workers, verbose, stats_dict, latched)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=2.0)
            if node.is_finished():
                break

    except Exception:
        print(f"Exception in publisher for topic '{topic_name}':")
        print(traceback.format_exc())

@typechecked
def _run_clock_publisher_process(libtype: ROSMsgLibType, start_sim_time: float,
                                  stop_event, all_done_event,
                                  stats_dict: Union[DictProxy, None] = None) -> None:
    """
    Publishes ``/clock`` messages, advancing simulated time in lock-step with
    wall time from ``start_sim_time``.  Runs until ``all_done_event`` is set
    (all data publishers finished) or ``stop_event`` is set (KeyboardInterrupt).
    """

    def _update_clock_stats(count: int, pub_start: float, prev_time: list) -> int:
        """Compute and write avg/inst hz and tick count into stats_dict. Returns updated count."""
        if stats_dict is None:
            return count + 1
        now = time.monotonic()
        count += 1
        elapsed = now - pub_start
        avg_hz = count / elapsed if elapsed > 0 else 0.0
        inst_hz = 1.0 / (now - prev_time[0]) if (now - prev_time[0]) > 0 else 0.0
        prev_time[0] = now
        entry = stats_dict[CLOCK_TOPIC]
        entry['last_update_time'] = now
        entry['progress'] = count
        entry['avg_hz'] = avg_hz
        entry['inst_hz'] = inst_hz
        stats_dict[CLOCK_TOPIC] = entry
        return count

    def _compute_sec_nsec() -> Tuple[int, int]:
        """Compute simulated time (sec, nsec) based on elapsed wall time."""
        elapsed = max(0.0, time.monotonic() - wall_start)
        sim_time = start_sim_time + elapsed
        sec = int(sim_time)
        nsec = int((sim_time - sec) * 1e9)
        return sec, nsec

    try:
        wall_start = time.monotonic() + 1.0  # 1-second ramp-up matches data publishers

        if libtype == ROSMsgLibType.ROSPY:
            import rospy
            Clock = ModuleImporter.get_module_attribute('rosgraph_msgs.msg', 'Clock')
            rospy.init_node("robotdataprocess_clock_publisher", anonymous=True)
            pub = rospy.Publisher(CLOCK_TOPIC, Clock, queue_size=10)

            count = 0
            pub_start = wall_start
            prev_time = [wall_start]
            while not rospy.is_shutdown() and not all_done_event.is_set() and not stop_event.is_set():
                sec, nsec = _compute_sec_nsec()
                msg = Clock()
                msg.clock = rospy.Time(secs=sec, nsecs=nsec)
                pub.publish(msg)
                count = _update_clock_stats(count, pub_start, prev_time)
                time.sleep(1.0 / CLOCK_HZ)

        elif libtype == ROSMsgLibType.RCLPY:
            import rclpy
            from rclpy.node import Node
            from rclpy.clock import Clock as RclpyClock, ClockType
            Clock = ModuleImporter.get_module_attribute('rosgraph_msgs.msg', 'Clock')

            class _ClockNode(Node):
                def __init__(self_node):
                    super().__init__("robotdataprocess_clock_publisher")
                    self_node._pub = self_node.create_publisher(Clock, CLOCK_TOPIC, 10)
                    # Use a steady (wall) clock so the timer is not blocked by use_sim_time
                    self_node._timer = self_node.create_timer(
                        1.0 / CLOCK_HZ, self_node._cb,
                        clock=RclpyClock(clock_type=ClockType.STEADY_TIME))
                    self_node._finished = False
                    self_node._count = 0
                    self_node._pub_start = wall_start
                    self_node._prev_time = [wall_start]

                def _cb(self_node):
                    if all_done_event.is_set() or stop_event.is_set():
                        self_node._timer.cancel()
                        self_node._finished = True
                        return
                    sec, nsec = _compute_sec_nsec()
                    msg = Clock()
                    msg.clock.sec = sec
                    msg.clock.nanosec = nsec
                    self_node._pub.publish(msg)
                    self_node._count = _update_clock_stats(self_node._count, self_node._pub_start, self_node._prev_time)

            if not rclpy.ok():
                rclpy.init()
            node = _ClockNode()
            while rclpy.ok() and not node._finished:
                rclpy.spin_once(node, timeout_sec=0.1)

    except Exception:
        print("Exception in clock publisher:")
        print(traceback.format_exc())


@typechecked
def publish_data_ROS_multiprocess(data_list: List[SequentialData], data_topics: List[str], data_msg_type: List[Union[str, None]],
                                  data_hz: List[float], data_workers: List[int], libtype: ROSMsgLibType, shutdown_ros: bool,
                                  verbose: bool = True, delay_seconds: float = 0.0, wait_for_sub: bool = False,
                                  publish_clock: bool = False, data_latched: Union[List[bool], None] = None) -> None:
    """
    Launches one publisher process per Data stream, either for ROS1 or ROS2.

    Args:
        data_list: list of Data objects
        data_topics: list of ROS topic names
        data_msg_type: list of ROS message types for data that can be published as multiple types.
        data_hz: The expected hertz of the data stream (used to set hertz speed of publishing process).
        data_workers: Each topic will have one publisher process and X number of worker processes to pull data.
        libtype: ROSMsgLibType indicating whether to use rospy (ROS1) or rclpy (ROS2).
        shutdown_ros: Whether to shutdown ROS after publishing is complete.
        verbose: Whether to print topic publishing status to the console.
        delay_seconds: Delay in seconds before doing anything (for testing purposes).
        wait_for_sub: Whether to wait for a subscriber before we start publishing (for testing purposes).
        publish_clock: Whether to publish a ``/clock`` topic alongside the data streams.
        data_latched: Per-topic flag indicating whether the publisher should be latched (e.g. for
            static transforms). Latched publishers hold the node alive until all non-latched
            publishers finish. Defaults to all False.
    """

    # Optional delay before starting (for testing purposes)
    if delay_seconds > 0.0:
        time.sleep(delay_seconds)

    # Ensure we have matching data and topics
    assert len(data_list) == len(data_topics), "First four arguments need to have the same length!"
    assert len(data_list) == len(data_msg_type), "First four arguments need to have the same length!"
    assert len(data_list) == len(data_workers), "First four arguments need to have the same length!"

    # Default: latch topics that have only a single timestamp (e.g. static transforms)
    if data_latched is None:
        data_latched = [data.len() == 1 for data in data_list]
    assert len(data_list) == len(data_latched), "data_latched must have the same length as data_list!"

    # Create a shared dictionary across processes for statistics
    manager = Manager()
    stats_dict = manager.dict() # Shared across processes

    # Initialize stats_dict for each topic
    all_topics = list(data_topics)
    if publish_clock:
        all_topics.append(CLOCK_TOPIC)
    for topic in all_topics:
        stats_dict[topic] = {"last_update_time": 0, "progress": 0, "total": 0, "avg_hz": 0,
                             "inst_hz": 0, "skipped": 0, "log_queue": manager.list(),
                             'local_buf_size': 0, 'worker_queue_size': 0}

    # Launch the appropriate publisher processes
    processes: List[Process] = []
    topic_to_proc: dict = {}
    stop_event = Event()
    for data, topic, type, hertz, workers, latched in zip(data_list, data_topics, data_msg_type, data_hz, data_workers, data_latched):
        if libtype == ROSMsgLibType.RCLPY:
            pub_proc_func = _run_ROS2_publisher_process
        elif libtype == ROSMsgLibType.ROSPY:
            pub_proc_func = _run_ROS_publisher_process

        p = Process(target=pub_proc_func, args=(data, topic, type, hertz, stop_event, wait_for_sub, workers, verbose, stats_dict, latched))
        p.start()
        processes.append(p)
        topic_to_proc[topic] = p

    # Launch the clock publisher if requested (tracked separately — it stops after data publishers finish)
    clock_process: Union[Process, None] = None
    all_done_event = Event()
    if publish_clock:
        start_sim_time = min(float(data.timestamps[0]) for data in data_list)
        clock_process = Process(target=_run_clock_publisher_process,
                                args=(libtype, start_sim_time, stop_event, all_done_event, stats_dict))
        clock_process.start()
        topic_to_proc[CLOCK_TOPIC] = clock_process

    # Helper to build the table with a Status column
    def generate_table() -> Table:
        table = Table(title="ROS Publisher Dashboard", show_header=True, header_style="bold magenta")
        table.add_column("Status", justify="center", min_width=14)
        table.add_column("Topic", style="cyan")
        table.add_column("Progress", justify="right", min_width=13)
        table.add_column("Avg Hz", justify="right")
        table.add_column("Inst Hz", justify="right")
        table.add_column("Skipped Msgs", justify="right")
        table.add_column("Buffer Sizes", justify="right")

        for topic, s in stats_dict.items():
            proc = topic_to_proc[topic]
            row_style = None

            if proc.is_alive():
                if time.monotonic() - float(s["last_update_time"]) < 0.3:
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
                f"{s['local_buf_size']}/∞ | {s['worker_queue_size']}/{MSG_QUEUE_MAX_SIZE}",
                style=row_style
            )
        return table

    # Separate latched and non-latched publisher processes
    non_latched_procs = [p for p, l in zip(processes, data_latched) if not l]
    latched_procs = [p for p, l in zip(processes, data_latched) if l]

    # Wait for non-latched publishers to finish, then signal latched ones to stop
    try:
        if verbose:
            with Live(generate_table(), refresh_per_second=10, screen=False) as live:
                while any(p.is_alive() for p in non_latched_procs):
                    for topic in data_topics:
                        logs = stats_dict[topic]["log_queue"]
                        while len(logs) > 0:
                            msg = logs.pop(0)
                            live.console.print(f"[{topic}] {msg}", markup=False, style="")
                    live.update(generate_table())
                    time.sleep(0.1)
                for p in non_latched_procs:
                    p.join()

                # Non-latched publishers are done; signal latched publishers to shut down
                stop_event.set()
                for p in latched_procs:
                    p.join()
                live.update(generate_table())
        else:
            for p in non_latched_procs:
                p.join()

            # Non-latched publishers are done; signal latched publishers to shut down
            stop_event.set()
            for p in latched_procs:
                p.join()
    except KeyboardInterrupt:
        stop_event.set()
        for p in processes:
            p.join()

    # Signal the clock publisher to stop and wait for it
    if clock_process is not None:
        all_done_event.set()
        clock_process.join()

    manager.shutdown()

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
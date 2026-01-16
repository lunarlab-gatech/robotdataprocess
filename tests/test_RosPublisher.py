from copy import deepcopy
from multiprocessing import Process
import numpy as np
import os
from pathlib import Path
from robotdataprocess.data_types.Data import Data, ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.ModuleImporter import ModuleImporter
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
import time
from typing import Any
from numpy.typing import NDArray
import unittest

@unittest.skipIf(os.getenv("SKIP_ROS2_TESTS") == "True", "ROS2 not installed")
class TestRosPublisher(unittest.TestCase):

    def util_ROS2_test(self, data: Data, topic_class: Any, topic_name: str, 
                       msg_to_dict_fn: function, assert_data_dict_equal: function):
        
        rclpy = ModuleImporter.get_module('rclpy')
        Node = ModuleImporter.get_module_attribute('rclpy.node', 'Node')
        CvBridge = ModuleImporter.get_module_attribute('cv_bridge', 'CvBridge')

        # Create a ROS2 node to subscribe to the published topic
        class Listener(Node):
            def __init__(self):
                super().__init__('listener')
                self.bridge = CvBridge()
                self.subscription = self.create_subscription(topic_class, topic_name, self.callback, 10)
                self.msg_to_dict_fn = msg_to_dict_fn
                self.received = []
            
            # Generic callback function to save recieved messages
            def callback(self, msg):
                self.received.append(msg_to_dict_fn(msg))
                self.get_logger().info(f"Received msg {len(self.received)-1} at time {self.received[-1]['stamp']}")

        
        # Launch the publisher we initialize rclpy (otherwise rclpy.init breaks process forking for ROS2)
        p = Process(target=publish_data_ROS_multiprocess, args=([data], [topic_name], [None], [1], 
                                                                ROSMsgLibType.RCLPY, False, True, 0.0))
        p.start()

        # Initialize ROS2 and create the listener node
        rclpy.init()
        node = Listener()

        # Start listening here and meanwhile launch the publisher 
        try:
            start_time = time.time()
            timeout_sec = 5.0
            while rclpy.ok() and (time.time() - start_time) < timeout_sec:
                rclpy.spin_once(node, timeout_sec=0.1)

            # Make sure we recieved at least one message
            self.assertTrue(len(node.received) > 0)

            # Make sure data is correct for each message we recieved
            data_ts_floats = data.timestamps.astype(float)
            for j in range(len(node.received)):
                stamp = node.received[j]['stamp']
                matches = np.where(np.isclose(data_ts_floats, stamp, atol=1e-6))[0]
                if matches.size > 0:
                    # Ensure that data matches the recieved msg dictionary
                    matched_index = matches[0]
                    assert_data_dict_equal(data, matched_index, node.received[j])
                else:
                    self.fail("Recieved ROS Message has a timestamp that doesn't match any of the sent messages!")

        finally:
            # ---------- Guaranteed cleanup ----------
            node.destroy_node()
            rclpy.shutdown()

        
    def test__run_ROS2_publisher_process(self):
        """ Test that we can publish to ROS2 without losing data."""

        # Create an ImageDataInMemory object
        file_path = Path(Path('.'), 'tests', 'files', 'test_RosPublisher', 'test__run_ROS2_publisher_process').absolute()
        image_data = ImageDataInMemory.from_image_files(file_path, '/cam0')

        # Lazily import ROS2 libraries
        Image = ModuleImporter.get_module_attribute('sensor_msgs.msg', 'Image')

        # Write msg_to_dict function
        def msg_to_dict_fn(msg: Image) -> dict:
            try:
                image: NDArray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                return {
                    "image": image,
                    "height": msg.height,
                    "width": msg.width,
                    "encoding": msg.encoding,
                    "frame_id": msg.header.frame_id,
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                }
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")

        # Write assert_data_dict_equal function
        def assert_data_dict_equal(data_sent: ImageData, matched_index: int, msg_recieved: dict):
            np.testing.assert_array_equal(data_sent.images[matched_index], msg_recieved["image"])
            np.testing.assert_equal(data_sent.height, msg_recieved["height"])
            np.testing.assert_equal(data_sent.width, msg_recieved["width"])
            np.testing.assert_equal(data_sent.encoding, ImageData.ImageEncoding.from_ros_str(msg_recieved["encoding"]))
            np.testing.assert_equal(data_sent.frame_id, msg_recieved["frame_id"])
            np.testing.assert_almost_equal(float(data_sent.timestamps[matched_index]), msg_recieved["stamp"])

        # Test that we can send data over ROS2 and get it back successfully
        self.util_ROS2_test(image_data, Image, '/cam0/image_raw', msg_to_dict_fn, assert_data_dict_equal)
 
if __name__ == "__main__":
    unittest.main()
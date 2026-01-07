from copy import deepcopy
import cv2
from decimal import Decimal
from multiprocessing import Process
import numpy as np
import os
from pathlib import Path
from robotdataprocess import CoordinateFrame
from robotdataprocess.data_types.Data import ROSMsgLibType
from robotdataprocess.data_types.ImageData.ImageData import ImageData
from robotdataprocess.data_types.ImageData.ImageDataInMemory import ImageDataInMemory
from robotdataprocess.ros.Ros2BagWrapper import Ros2BagWrapper
from robotdataprocess.ros.RosPublisher import publish_data_ROS_multiprocess
import time
from numpy.typing import NDArray
import unittest

@unittest.skipIf(os.getenv("SKIP_ROS2_TESTS") == "True", "ROS2 not installed")
class TestRosPublisher(unittest.TestCase):
    
    def test__run_ROS2_publisher_process(self):
        """ Test that we can publish to ROS2 without losing data."""

        # Create an ImageDataInMemory object
        file_path = Path(Path('.'), 'tests', 'files', 'test_RosPublisher', 'test__run_ROS2_publisher_process').absolute()
        image_data = ImageDataInMemory.from_image_files(file_path, '/cam0')

        # Lazily import ROS2 libraries
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        # Create a ROS2 node to subscribe to the published topic
        class ImageListener(Node):
            def __init__(self):
                super().__init__('image_listener')
                self.bridge = CvBridge()
                self.subscription = self.create_subscription(Image, '/cam0/image_raw', self.image_callback, 10)
                self.received = []

            def image_callback(self, msg: Image):
                try:
                    # Convert ROS Image message to OpenCV image
                    image: NDArray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    self.received.append({
                        "image": image,
                        "height": msg.height,
                        "width": msg.width,
                        "encoding": msg.encoding,
                        "frame_id": msg.header.frame_id,
                        "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    })
                    self.get_logger().info(f"Received image {len(self.received)-1} at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")

                except Exception as e:
                    self.get_logger().error(f"Failed to convert image: {e}")

        # Launch the publisher we initialize rclpy (otherwise rclpy.init breaks process forking for ROS2)
        p = Process(target=publish_data_ROS_multiprocess, args=([image_data], ['/cam0/image_raw'], [None], ROSMsgLibType.RCLPY, False, False, 5.0))
        p.start()

        # Initialize ROS2 and create the listener node
        rclpy.init()
        node = ImageListener()

        # Start listening here and meanwhile launch the publisher 
        try:
            start_time = time.time()
            timeout_sec = 15.0
            while rclpy.ok() and (time.time() - start_time) < timeout_sec:
                rclpy.spin_once(node, timeout_sec=0.5)

            # Extract the recieved images for comparison
            np.testing.assert_equal(image_data.len(), len(node.received))
            for i in range(image_data.len()):
                np.testing.assert_array_equal(image_data.images[i], node.received[i]["image"])
                np.testing.assert_equal(image_data.height, node.received[i]["height"])
                np.testing.assert_equal(image_data.width, node.received[i]["width"])
                np.testing.assert_equal(image_data.encoding, ImageData.ImageEncoding.from_ros_str(node.received[i]["encoding"]))
                np.testing.assert_equal(image_data.frame_id, node.received[i]["frame_id"])
                np.testing.assert_almost_equal(float(image_data.timestamps[i]), node.received[i]["stamp"])

        finally:
            # ---------- Guaranteed cleanup ----------
            node.destroy_node()
            rclpy.shutdown()
        
if __name__ == "__main__":
    unittest.main()
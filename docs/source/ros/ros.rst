ROS Integration
===============

robotdataprocess can publish any ``SequentialData`` object directly to ROS1 or
ROS2 topics without converting to bag files first. It can also read and write
ROS2 bag files via the ``Ros2BagWrapper``.

Publishing Data
---------------

``publish_data_ROS_multiprocess`` launches one publisher process per data stream.
Worker processes pre-build ROS messages into a queue while the main thread
publishes them at the correct timestamps.

.. autofunction:: robotdataprocess.publish_data_ROS_multiprocess

Key points:

* Set ``libtype`` to ``ROSMsgLibType.RCLPY`` for ROS2 or ``ROSMsgLibType.ROSPY`` for ROS1.
* For ROS1, ensure ``roscore`` is running before calling this function.
* ``data_msg_type`` allows selecting alternate message types (e.g. ``"Path"`` instead of ``"Odometry"`` for OdometryData). Pass ``None`` to use the default.
* ``data_workers`` controls parallelism: each topic gets one publisher process and N worker processes that pre-build messages.

Ros2BagWrapper
--------------

.. autoclass:: robotdataprocess.Ros2BagWrapper
   :members:
   :show-inheritance:

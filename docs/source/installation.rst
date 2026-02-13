Installation
============

Requirements
------------

**robotdataprocess** supports:

* **Python 3.8** -- for use with ROS1 Noetic and ROS2 Foxy/Galactic
* **Python 3.10+** -- for use with ROS2 Humble and later

ROS is **not required** for pure-Python workflows (loading, manipulating, visualizing, and saving data). ROS dependencies are lazily imported only when publishing to ROS topics.

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/lunarlab-gatech/robotdataprocess.git
   cd robotdataprocess
   git submodule init
   git submodule update
   pip install .

ROS Setup (Optional)
--------------------

To publish data over ROS topics, you need a working ROS installation:

* **ROS2**: Install `rclpy <https://docs.ros.org/en/humble/Installation.html>`_ and the relevant message packages (``std_msgs``, ``sensor_msgs``, ``nav_msgs``, ``geometry_msgs``).
* **ROS1**: Install `rospy <http://wiki.ros.org/noetic/Installation>`_ (Noetic) and the same message packages. Ensure ``roscore`` is running before publishing.

Docker (ROS1)
-------------

For ROS1 testing in Docker, use shared memory to support the multiprocess publisher:

.. code-block:: bash

   docker run --shm-size=2gb ...

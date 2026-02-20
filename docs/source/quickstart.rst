Quick Start
===========

Core Concepts
-------------

All sensor data is represented by **Data objects** that provide common base methods for loading, manipulating, visualizing, and exporting data:

* **Load** from any supported format (ROS2 bag, CSV, TXT, PNG folder, etc.)
* **Manipulate** (crop, shift, convert frames, interpolate)
* **Visualize** (matplotlib plots)
* **Export** (save to file, publish over ROS1/ROS2)

The class hierarchy is:

::

   Data
   ├── SequentialData          (adds timestamps)
   │   ├── PathData            (positions + orientations)
   │   │   └── OdometryData    (adds child_frame_id, ROS Odometry messages)
   │   ├── ImuData             (linear acceleration + angular velocity)
   │   ├── LiDARData           (point clouds)
   │   └── ImageData           (camera images)
   │       ├── ImageDataInMemory
   │       └── ImageDataOnDisk
   └── LoopClosureData         (inter-robot loop closure constraints)

Two enums control behavior throughout:

* ``CoordinateFrame``: ``FLU``, ``NED``, ``ENU``, ``NONE`` -- tracks which coordinate convention a Data object uses, enabling built-in frame conversions (e.g. ``to_coordinate_frame()``) that correctly transform positions and orientations.
* ``ROSMsgLibType``: ``ROSBAGS`` (rosbags, pure Python), ``RCLPY`` (ROS2), ``ROSPY`` (ROS1), ``NONE`` -- specifies which ROS message library to use when building or serializing ROS messages, so the same Data object can target different ROS environments.

Loading Data
------------

Each data type provides class methods for loading from various formats:

.. code-block:: python

   from robotdataprocess import OdometryData, ImuData, LiDARData, CoordinateFrame
   from robotdataprocess import ImageDataOnDisk, ImageDataInMemory

   # Odometry from CSV (timestamp, x, y, z, qw, qx, qy, qz)
   odom = OdometryData.from_csv("odom.csv", "world", "base_link",
                                 CoordinateFrame.FLU, header_included=True)

   # Odometry from ROS2 bag
   odom = OdometryData.from_ros2_bag("path/to/bag", "/odom", CoordinateFrame.FLU)

   # IMU from TXT file (TartanAir format)
   imu = ImuData.from_txt_file("imu.txt", "imu_link", CoordinateFrame.NED)

   # LiDAR from .npy files (one per scan, filenames are timestamps)
   lidar = LiDARData.from_npy_files("scans/", "velodyne", CoordinateFrame.FLU)

   # Images from PNG folder (filenames are timestamps)
   images = ImageDataOnDisk.from_image_files("images/", "camera")

Manipulating Data
-----------------

.. code-block:: python

   from decimal import Decimal

   # Crop to a time window
   odom.crop_data(Decimal("100.0"), Decimal("200.0"))

   # Convert coordinate frames (NED -> FLU)
   odom.to_coordinate_frame(CoordinateFrame.FLU)

   # Shift all positions
   odom.shift_position(x_shift=1.0, y_shift=0.0, z_shift=0.0)

   # Move the first pose to the origin
   odom.shift_to_start_at_identity()

   # Resample to a target frequency
   odom.interpolate_to_hz(50.0)

   # Round timestamps (useful before comparison)
   odom.round_timestamps(decimals=3)

Visualization
-------------

.. code-block:: python

   from robotdataprocess import PathData

   # 2D trajectory plot (saves to PDF)
   PathData.visualize_2D(
       [est_path, gt_path],
       [False, True],                  # isGTList
       ["#FF0000", "#0000FF"],          # colors
       ["Estimated", "Ground Truth"],   # names
       save_path="trajectories.pdf",
   )

   # 3D trajectory plot with orientation axes
   est_path.visualize_3D([gt_path], ["Estimated", "Ground Truth"])

   # IMU data plots
   imu.visualize(ts_start=0.0, ts_end=100.0)

   # Hertz analysis (timing histograms)
   odom.hertz_analysis()

Trajectory Evaluation
---------------------

.. code-block:: python

   # Align trajectories and compute APE/RPE metrics via the evo library
   results, est_aligned, gt_aligned = PathData.align_and_calculate_traj_errors(
       gt_path, est_path, max_diff=0.1
   )

   # Access specific metrics
   ape_rmse = results["APE"]["translation_part"]["rmse"]
   rpe_mean = results["RPE"]["rotation_angle_deg"]["mean"]

Publishing over ROS
-------------------

No bag conversion needed -- publish data directly to ROS topics:

**ROS2 (rclpy):**

.. code-block:: python

   from robotdataprocess import publish_data_ROS_multiprocess, ROSMsgLibType

   publish_data_ROS_multiprocess(
       data_list=[odom, imu],
       data_topics=["/odom", "/imu"],
       data_msg_type=[None, None],        # None uses the default message type
       data_hz=[100, 500],                # expected frequency per topic
       data_workers=[1, 1],               # worker processes per topic
       libtype=ROSMsgLibType.RCLPY,
       shutdown_ros=True,
       verbose=True,
   )

**ROS1 (rospy):**

.. code-block:: python

   from robotdataprocess import publish_data_ROS_multiprocess, ROSMsgLibType

   # Ensure roscore is running before calling this
   publish_data_ROS_multiprocess(
       data_list=[odom, imu],
       data_topics=["/odom", "/imu"],
       data_msg_type=[None, None],
       data_hz=[100, 500],
       data_workers=[1, 1],
       libtype=ROSMsgLibType.ROSPY,
       shutdown_ros=True,
       verbose=True,
   )

Exporting Data
--------------

.. code-block:: python

   # Save odometry to CSV
   odom.to_csv("output.csv")

   # Save images to .npy (memory-mapped)
   images.to_npy("output_folder/")

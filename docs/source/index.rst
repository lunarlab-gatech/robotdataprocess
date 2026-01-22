.. robotdataprocess documentation master file, created by
   sphinx-quickstart on Thu Dec 11 15:27:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

robotdataprocess
============================================

.. container:: badges

   .. image:: https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml/badge.svg?branch=master
      :target: https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml
      :alt: Python Unit Tests

   .. image:: https://coveralls.io/repos/github/lunarlab-gatech/robotdataprocess/badge.svg?branch=master
      :target: https://coveralls.io/github/lunarlab-gatech/robotdataprocess?branch=master
      :alt: Coverage Status

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Code Documentation

   data_types/data_types.rst


**robotodataprocess** is a Python library for loading, manipulating, saving, publishing, and evaluating multi-robot SLAM datasets. Mapping and Localization have a variety of attributes that lead to difficulty for research and development:

1. Algorithms are written using different environments and ROS versions. Dataset processing code written for one rarely works (without modifications) in another.
2. Datasets are found in different coordinate frames and formats, requiring specialized code for each one.
3. The sheer amount of necessary code often written to accomplish the above makes experiment verification and debugging more difficult.

This library was written to tackle these issues. **robotdataprocess** is fully supported in:

* Python 3.8 (ROS1 Noetic & ROS2 Foxy/Galactic)
* Python 3.10 (Ros2 Humble and later)

Additionally, data is loaded into ``Data`` objects that are agnostic to the original input format or the desired output (either saved or published over ROS). Thus, working with a new dataset format only requires writing a simple dataloading function, and all other features (manipulation, visualization, export) will work automatically. Similarly, exporting to a new format only requires writting a new export function. This design also leads to intutive high-level coding that abstracts away the lower level details and speeds up integration. For example, here is the code to load IMU data and publish it over ROS2:

.. code-block:: python
   
   # Load IMU data from txt file
   imu_data = ImuData.from_txt_file(file_path, frame_id, CoordinateFrame.NED)

   # Convert data from NED frame to FLU frame
   pose_data.to_FLU_frame()

   # Crop the data
   imu_data.crop_data(Decimal('0.0'), end_time) 

   # Publish it over ROS2
   publish_data_ROS_multiprocess([imu_data], ['/imu0'], [None], [500], [1], 
                                  ROSMsgLibType.RCLPY, True, verbose=True)

Finally, most functionality in **robotdataprocess** is covered with extensive test cases, build status on supported versions is verified with GitHub Actions, and code coverage is reported using Coveralls. Thus, researchers can be confident that data is in the proper format for downstream application on odometry and multi-robot SLAM algorithms.

At the `Lunar Lab <https://sites.gatech.edu/lunarlab/>`_, this library has been used to generate rosbags, publish to ROS topics, or evaluate output trajectories with each of the following repositories:

- `VINS-Mono <https://github.com/lunarlab-gatech/VINS-MONO-ROS2>`_
- `OpenVINS <https://github.com/lunarlab-gatech/open_vins/tree/docker>`_
- `LIO-SAM <https://github.com/lunarlab-gatech/LIO-SAM>`_
- maplab
- ROMAN
- SlideSLAM

For examples of how to use this repository, see the ``examples`` folder. For information on working with a particular data type, click on the Data type in the navigation bar on the left.

Contact Us
==========
For questions or inquiries, feel free to reach out to the authors:

- `Daniel Butterfield <https://sites.gatech.edu/lunarlab/members/>`_
- `Lunar Lab <https://sites.gatech.edu/lunarlab/>`_
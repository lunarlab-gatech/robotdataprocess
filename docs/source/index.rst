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
   :caption: Getting Started

   installation.rst
   quickstart.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   data_types/data_types.rst
   ros/ros.rst


**robotdataprocess** is a Python library for loading, manipulating, saving, publishing, and evaluating multi-robot SLAM datasets. It provides a unified interface across ROS versions, coordinate frames, and file formats, eliminating the per-dataset boilerplate that slows down robotics research.

Supported environments:

* **Python 3.8** -- ROS1 Noetic, ROS2 Foxy/Galactic
* **Python 3.10+** -- ROS2 Humble and later

All sensor data is loaded into format-agnostic ``Data`` objects. Adding support for a new input format requires only a dataloader; all manipulation, visualization, and export methods work automatically. ROS is not required for pure-Python workflows.

.. code-block:: python

   from robotdataprocess import ImuData, CoordinateFrame, ROSMsgLibType
   from robotdataprocess import publish_data_ROS_multiprocess
   from decimal import Decimal

   imu = ImuData.from_txt_file(path, "imu_link", CoordinateFrame.NED)
   imu.to_FLU_frame()
   imu.crop_data(Decimal("0.0"), end_time)

   publish_data_ROS_multiprocess(
       [imu], ["/imu0"], [None], [500], [1],
       ROSMsgLibType.RCLPY, True, verbose=True,
   )

The library is tested across all supported Python versions via GitHub Actions, with coverage reported through Coveralls.

At the `Lunar Lab <https://sites.gatech.edu/lunarlab/>`_, robotdataprocess has been used with `VINS-Mono <https://github.com/lunarlab-gatech/VINS-MONO-ROS2>`_, `OpenVINS <https://github.com/lunarlab-gatech/open_vins/tree/docker>`_, `LIO-SAM <https://github.com/lunarlab-gatech/LIO-SAM>`_, and `ROMAN <https://github.com/mit-acl/roman>`_.

See the ``examples`` folder for usage examples, or browse the data types and ROS integration pages in the sidebar.

Contact Us
==========
For questions or inquiries, feel free to reach out to the authors:

- `Daniel Butterfield <https://sites.gatech.edu/lunarlab/members/>`_
- `Lunar Lab <https://sites.gatech.edu/lunarlab/>`_
Data Types
==========

All data classes inherit from ``Data``, which provides a ``frame_id`` attribute.
Time-ordered data extends ``SequentialData``, adding timestamps and hertz analysis.
Spatial pose data extends ``PathData``, adding positions, orientations, and a coordinate frame.

Enumerations
------------

.. autoclass:: robotdataprocess.CoordinateFrame
   :members:
   :undoc-members:

.. autoclass:: robotdataprocess.ROSMsgLibType
   :members:
   :undoc-members:

SequentialData
--------------

.. autoclass:: robotdataprocess.SequentialData
   :members:
   :show-inheritance:

PathData
--------

.. autoclass:: robotdataprocess.PathData
   :members:
   :show-inheritance:

OdometryData
-------------

.. autoclass:: robotdataprocess.OdometryData
   :members:
   :show-inheritance:

ImuData
-------

.. autoclass:: robotdataprocess.ImuData
   :members:
   :show-inheritance:

LiDARData
---------

.. autoclass:: robotdataprocess.data_types.LiDARData.LiDARData
   :members:
   :show-inheritance:

LoopClosureData
---------------

.. autoclass:: robotdataprocess.LoopClosureData
   :members:
   :show-inheritance:

ImageData
---------

.. autoclass:: robotdataprocess.data_types.ImageData.ImageData.ImageData
   :members:
   :show-inheritance:

ImageDataInMemory
^^^^^^^^^^^^^^^^^

.. autoclass:: robotdataprocess.ImageDataInMemory
   :members:
   :show-inheritance:

ImageDataOnDisk
^^^^^^^^^^^^^^^

.. autoclass:: robotdataprocess.ImageDataOnDisk
   :members:
   :show-inheritance:

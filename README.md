# robotdataprocess

[![Python Unit Tests](https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml/badge.svg?branch=master)](https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml) [![Coverage Status](https://coveralls.io/repos/github/lunarlab-gatech/robotdataprocess/badge.svg?branch=master)](https://coveralls.io/github/lunarlab-gatech/robotdataprocess?branch=master)

A library for loading, saving, converting, and manipulating robotic datasets.

Currently, for operations with rosbags, the library inputs and outputs ROS2 Humble bags. For converting to ROS1 bags, the output bag will be compatible with ROS1 Noetic.

**WARNING:** Currently, this repository is in active development and functionality isn't guaranteed to work. If you will depend on this repository for important tasks, perhaps write test cases for the corresponding functionality before deployment/use.

## Installation

You will need Python 3.8+ to run this code. Run the following commands to install the repository:
```
git submodule init
git submodule update
pip install .
```

## Example Use

### Command Line Examples

See use cases for this repository below. For more information on specific operations, see the corresponding config file. All of these operations can be launched by running the following command below (or a similar one):
```
robotdataprocess <repository_path>/config/<operation name.yaml
```

#### Hertz Analysis
Generate histograms with an analysis of the hertz rates for a specific topic. If you want to investigate the `/tf` or `/tf_static` topic, these contain headers per transform, and thus you need to specify the name of the robot, as shown in `config/hertz_analysis_tf.yaml`.

#### View IMU Data
Plot the linear acceleration, angular velocity, and orientation data of an IMU topic in a ROS2 bag.

#### Downsample (& Prune)

This functionality allows downsampling the hertz rates of topics in the bag (for example, 40 Hz to 5 Hz). It can also prune topics by setting `include_unmentioned_topics` to false. 

#### Crop
A ROS2 bag can be cropped to only include message written into the bag within a certain timeframe with this feature. Note that this doesn't take into account the timestamps inside the messages, but just the recieve time of the messages.

#### ROS2 to ROS1 conversion
A ROS2 bag can be converted to ROS1 with this functionality, note that only certain message types are supported (see `msg_mapping_ros2_to_ros1` in `Ros2BagWrapper.py` for all supported messages). When running the command, any unsuported message types will print a warning message. 

Note that support for more messages can be easily added by altering `get_mapping()` and `msg_mapping_ros2_to_ros1` in `Ros2BagWrapper.py`.

#### Extract Odometry to CSV
Extract odometry from a ROS2 bag and save in a csv file.

#### Extract Images to Npy
Extract images from a ROS2 bag and saves them into .npy files. 

#### Compare Timestamps between two sets of Image Data
Compare timestamps between two sets of image data, loaded from .npy files.

### Code Examples

As robotic data can be saved in a variety of formats, the code structure is transitioning towards data objects that represent a data type. Thus, a data type can be loaded from a variety of formats, manipulated or visualized in various ways, and then exported into a new format.

Various examples of doing this can be seen in the `examples` directory. For any specific type of data, see its corresponding data class in the `src/robotdataprocess/data_types` directory.

For example, `OdometryData` can:
- Load from ROS2 bag, CSV file, or TXT file.
- Add noise or shift position.
- Export to ROS2 bag or CSV file.
- Visualize various OdometryData classes as paths via matplotlib.
- Convert from NED to ROS frame.

## Validation

### Unit Tests & Coverage

Run the following command to run the unit tests and generate a code coverage report:
```
coverage run --source robotdataprocess -m unittest discover tests/ -v
coverage report
coverage html
```

### Profiling

Run the following command to profile the code (via the unit tests):
```
python -m cProfile -o profile.out -m unittest discover tests/ -v
snakeviz profile.out
```
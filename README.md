# robotdataprocess

[![Python Unit Tests](https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml/badge.svg?branch=master)](https://github.com/lunarlab-gatech/robotdataprocess/actions/workflows/python_test.yml) [![Coverage Status](https://coveralls.io/repos/github/lunarlab-gatech/robotdataprocess/badge.svg?branch=master)](https://coveralls.io/github/lunarlab-gatech/robotdataprocess?branch=master)

A library for loading, saving, converting, publishsing, and manipulating robotic datasets. Most notably, it can load data in a variety of formats and then publish them live over ROS1 or ROS2. This circumvents the need to convert datasets into rosbags first, saving disk space.

**WARNING:** Currently, this repository is in active development and functionality isn't guaranteed to work. If you will depend on this repository for important tasks, perhaps write test cases for the corresponding functionality before deployment/use.

## Installation

This repository is officially supported with:
- Python 3.8 (for use with ROS1 Noetic & ROS2 Foxy/Galactic) 
- Python 3.10 (for use with ROS2 Humble and later)

Run the following commands to install the repository:
```
git submodule init
git submodule update
pip install .
```

## Documentation

Documentation is a WIP.

### Code Examples

As robotic data can be saved in a variety of formats, the code structure is transitioning towards data objects that represent a data type. Thus, a data type can be loaded from a variety of formats, manipulated or visualized in various ways, and then exported into a new format.

Various examples of doing this can be seen in the `examples` directory. For any specific type of data, see its corresponding data class in the `src/robotdataprocess/data_types` directory.

For example, `OdometryData` can:
- Load from ROS2 bag, CSV file, or TXT file.
- Add noise or shift position.
- Export to ROS2 bag, publish over ROS2 or ROS1, or save to a CSV file.
- Visualize various OdometryData classes as paths via matplotlib.
- Convert frames.

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
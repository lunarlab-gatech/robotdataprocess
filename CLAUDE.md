# CLAUDE.md - robotdataprocess

## Project Overview
Python library for loading, saving, converting, publishing, and manipulating robotic datasets. Supports both ROS1 (rospy/noetic) and ROS2 (rclpy). Built with hatchling.

## Repository Structure
```
src/robotdataprocess/
  data_types/          # Core data classes
    Data.py            # Base class, CoordinateFrame & ROSMsgLibType enums
    SequentialData.py  # Base for time-ordered data (timestamps, hertz)
    ImuData.py         # IMU sensor data
    LiDARData.py       # Point cloud data
    PathData.py        # Trajectory/path base class (extends SequentialData)
    OdometryData.py    # Odometry/pose data (extends PathData)
    LoopClosureData.py # Loop closure constraints
    ImageData/         # ImageData base, ImageDataInMemory, ImageDataOnDisk
  ros/
    RosPublisher.py    # Multiprocess ROS publisher (_SingleDataPublisher, publish_data_ROS_multiprocess)
    Ros2BagWrapper.py  # ROS2 bag file reading/writing/conversion
  ModuleImporter.py    # Lazy import + caching for ROS modules
tests/
  files/               # Test fixtures organized by test class name
  test_*.py            # unittest-based test files (13 total)
```

## Build & Install
```bash
pip install -e .
```

## Running Tests
Uses `unittest` (no pytest). No conftest.py.

```bash
# Run all tests
python3 -m unittest discover tests -v

# Skip subsets via environment variables
export SKIP_PURE_PYTHON_TESTS=True   # Skip non-ROS tests
export SKIP_ROS2_TESTS=True          # Skip ROS2 tests
export SKIP_ROS1_TESTS=True          # Skip ROS1 tests

# Run a specific test class
python3 -m unittest tests.test_RosPublisher.TestRosPublisherROS1 -v

# Coverage (multiprocessing-aware, see .coveragerc)
coverage run -m unittest discover tests
```

## ROS Test Requirements
- **ROS1 tests** (`TestRosPublisherROS1`): Require ROS1 noetic. The test `setUpClass` auto-starts roscore if not running. Tests use deeply nested multiprocessing (test -> util process -> publish_data_ROS_multiprocess -> publisher process -> worker processes). Each level may call `rospy.init_node`.
- **ROS2 tests** (`TestRosPublisher`): Require rclpy. Test wraps each sub-test in its own `Process` to isolate `rclpy.init`/`rclpy.shutdown`.
- Both test classes wrap `util_ROS*_test` in a child `Process` to isolate ROS node lifecycle.

## Key Patterns
- `@typechecked` decorator (typeguard) used extensively on public methods
- `ModuleImporter` for lazy-loading ROS dependencies so pure-Python usage works without ROS installed
- `ROSMsgLibType` enum: ROSBAGS (pure python), RCLPY (ROS2), ROSPY (ROS1), NONE (testing)
- Data classes have `get_ros_msg()` and `get_ros_msg_type()` for ROS message conversion
- RosPublisher uses shared memory optimization for ROS1 (disabled for ROS2)
- Worker processes pre-build ROS messages into a queue; main thread publishes based on timestamps

## Git Workflow
- This CLAUDE.md lives on the `develop` branch
- `master` is the main/stable branch; PRs target `master`
- `develop` is the active development branch

## Docker
- ROS1 container: Ubuntu 20.04, Python 3.8, ROS noetic
- Container name pattern: `robotdataprocess_ros1_container`
- Shared memory may be limited in Docker (affects RosPublisher shared_memory); use `--shm-size=2gb`

## Common Pitfalls
- ROS1 tests freeze if roscore is not running (`rospy.init_node` blocks indefinitely)
- Process joins in ROS tests need timeouts to prevent infinite hangs
- `neutralize_resource_tracker()` in RosPublisher.py patches Python's shared memory resource tracker to avoid crashes in multiprocessing
- `signal.SIGALRM` is used in worker processes to timeout shared memory allocation

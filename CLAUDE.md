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

## Coverage
- `.coveragerc` has `concurrency = multiprocessing`, so after `coverage run` you **must** run `coverage combine` before `coverage report`. Without `combine`, subprocess coverage data is not merged and the report will be stale.
- Full sequence: `coverage run -m unittest discover tests && coverage combine && coverage report`

## Writing Tests
- Tests use `unittest` exclusively (no pytest fixtures, parametrize, etc.)
- For tests that render matplotlib plots, set the Agg backend **before** importing any module that imports matplotlib: `import matplotlib; matplotlib.use('Agg')` at the top of the test file, before data class imports.
- Test fixtures live in `tests/files/<TestClassName>/`. Helper data (CSV, TXT, etc.) goes there.

## Documentation (docs/)
- Sphinx with `sphinx_rtd_theme`, `autodoc`, and `napoleon` extensions. Config in `docs/source/conf.py`.
- Structure: `index.rst` (overview) → `installation.rst`, `quickstart.rst` (Getting Started) → `data_types/data_types.rst`, `ros/ros.rst` (API Reference).
- Pages use `.. autoclass::` / `.. autofunction::` directives to pull docstrings from source.
- Style preferences: concise and professional. Avoid verbose problem statements. Don't recommend editable (`-e`) installs. Don't reference `to_evo()` in docs.
- When describing enums: `CoordinateFrame` enables frame conversions (e.g. `to_FLU_frame()`); `ROSMsgLibType` defines which ROS message library to use. `ROSBAGS` should be described as "(rosbags, pure Python)".
- ROS publishing examples should show separate ROS1 and ROS2 code blocks.

## Common Pitfalls
- ROS1 tests freeze if roscore is not running (`rospy.init_node` blocks indefinitely)
- Process joins in ROS tests need timeouts to prevent infinite hangs
- `neutralize_resource_tracker()` in RosPublisher.py patches Python's shared memory resource tracker to avoid crashes in multiprocessing
- `signal.SIGALRM` is used in worker processes to timeout shared memory allocation
- OdometryData.py has several unused-import Pylance warnings (col_to_dec_arr, dec_arr_to_float_arr, geometry, plt, NDArray, R, timestamp) -- these are pre-existing and not regressions

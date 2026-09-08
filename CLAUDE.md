# CLAUDE.md - robotdataprocess

## Repository Specific Info
Python library for loading, saving, converting, publishing, and manipulating robotic datasets. Supports both ROS1 (rospy/noetic) and ROS2 (rclpy). Built with hatchling.

### Repository Structure
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

### Build & Install
```bash
pip install -e .
```

### Running Tests
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

### ROS Test Requirements
- **ROS1 tests** (`TestRosPublisherROS1`): Require ROS1 noetic. The test `setUpClass` auto-starts roscore if not running. Tests use deeply nested multiprocessing (test -> util process -> publish_data_ROS_multiprocess -> publisher process -> worker processes). Each level may call `rospy.init_node`.
- **ROS2 tests** (`TestRosPublisher`): Require rclpy. Test wraps each sub-test in its own `Process` to isolate `rclpy.init`/`rclpy.shutdown`.
- Both test classes wrap `util_ROS*_test` in a child `Process` to isolate ROS node lifecycle.

### Key Patterns
- `ModuleImporter` for lazy-loading ROS dependencies so pure-Python usage works without ROS installed
- `ROSMsgLibType` enum: ROSBAGS (pure python), RCLPY (ROS2), ROSPY (ROS1), NONE (testing)
- Data classes have `get_ros_msg()` and `get_ros_msg_type()` for ROS message conversion
- RosPublisher uses shared memory optimization for ROS1 (disabled for ROS2)
- Worker processes pre-build ROS messages into a queue; main thread publishes based on timestamps

### Coverage
- `.coveragerc` has `concurrency = multiprocessing`, so after `coverage run` you **must** run `coverage combine` before `coverage report`. Without `combine`, subprocess coverage data is not merged and the report will be stale.
- Full sequence: `coverage run -m unittest discover tests && coverage combine && coverage report`

### Writing Tests
- Tests use `unittest` exclusively (no pytest fixtures, parametrize, etc.)
- For tests that render matplotlib plots, set the Agg backend **before** importing any module that imports matplotlib: `import matplotlib; matplotlib.use('Agg')` at the top of the test file, before data class imports.
- Test fixtures live in `tests/files/<TestClassName>/`. Helper data (CSV, TXT, etc.) goes there.

### Documentation (docs/)
- Sphinx with `sphinx_rtd_theme`, `autodoc`, and `napoleon` extensions. Config in `docs/source/conf.py`.
- Structure: `index.rst` (overview) → `installation.rst`, `quickstart.rst` (Getting Started) → `data_types/data_types.rst`, `ros/ros.rst` (API Reference).
- Pages use `.. autoclass::` / `.. autofunction::` directives to pull docstrings from source.
- Style preferences: concise and professional. Avoid verbose problem statements. Don't recommend editable (`-e`) installs. Don't reference `to_evo()` in docs.
- When describing enums: `CoordinateFrame` enables frame conversions (e.g. `to_FLU_frame()`); `ROSMsgLibType` defines which ROS message library to use. `ROSBAGS` should be described as "(rosbags, pure Python)".
- ROS publishing examples should show separate ROS1 and ROS2 code blocks.

### Common Pitfalls
- ROS1 tests freeze if roscore is not running (`rospy.init_node` blocks indefinitely)
- Process joins in ROS tests need timeouts to prevent infinite hangs
- `neutralize_resource_tracker()` in RosPublisher.py patches Python's shared memory resource tracker to avoid crashes in multiprocessing
- `signal.SIGALRM` is used in worker processes to timeout shared memory allocation
- OdometryData.py has several unused-import Pylance warnings (col_to_dec_arr, dec_arr_to_float_arr, geometry, plt, NDArray, R, timestamp) -- these are pre-existing and not regressions


## Instructions for Writing Code

### Coding Principles
- Single Responsibility Principle (SRP) - Each class or function should have a single responsiblity, which leads
to better reusability, reduced complexity, and easier maintenance.

- Don't Repeat Yourself (DRY) - No copy-pasted code; As bugs found in the copy-pasted code is very difficult
to ensure are fixed in all locations. Constructing shared helpers to share functinoality or updating previous
code to support both (for example, upgrading a conversion fucntion to support a new conversion) allows us to
fulfill the DRY principle and also fulfills the SRP princple as the function still has one responsibility (image
Conversion).

### Coding Style

- Google-style docstrings. One line for simple functions; a full docstring with `Args`/`Returns`
  for larger ones. At most one paragraph of prose in a docstring (not counting `Args`/`Returns`
  themselves) except in genuinely unusual cases.
- Type hints on every parameter, return value, and non-obvious variable.
- No global variables anywhere in this library — they should be part of a class instead.
  Module-level `UPPER_CASE` constants are fine: they are not variables, and forcing tuning
  values or topic names into a class to satisfy the rule reads worse than leaving them.
- Inline comments are one line ~95% of the time; two-plus lines only in extreme cases.
-- Imports listed alphabetically (e.g. `import numpy`; `import robotdataprocess`). This means one
  flat list sorted case-insensitively by module path — not grouped into stdlib/third-party/local
  blocks (no isort-style grouping). A relative import (`from .foo import bar`) sorts by its bare
  module name (`foo`), ignoring the leading dot. Example:
  ```python
  from .air_museum import load_AirMuseum_data
  from MeronomyGraph.params.data_params import DataParams
  from pathlib import Path
  from robotdatapy.data.pose_data import PoseData
  from typing import Callable
  ```
- Type hints on non-obvious variables only apply to a line that also does something (an actual
  assignment or expression) — never add a bare `name: Type` declaration line with nothing else on
  it just to satisfy this rule. If a variable is assigned conditionally across branches, annotate
  the first assignment (or skip the annotation if it isn't genuinely needed for clarity).
- No file-level top-of-file comment block — put that content in the relevant class/function
  docstring instead. This applies to CLI-launcher-style files too: give them a `__main__` guard and
  put the file's explanation in its docstring, not a header comment.
- Every class attribute that gets set anywhere in the class must be declared up front in the class
  body with a type annotation — even though Python allows adding attributes on the fly elsewhere,
  don't rely on that; it makes classes hard to reason about at a glance. If a new value is set on
  the class in any of its functions, it needs to be added here too. Ideally, every attribute is
  initialized (or set to `None`) in `__init__`. Example:

  ```python
  @typechecked
  class OdometryData(PathData):
      """
      Odometry data extending PathData with a child frame ID and ROS message caching.

      Supports loading from ROS2 bags, CSV files, and TXT files, and exporting
      to CSV or ROS messages (Odometry, Path, and maplab OdometryWithImuBiases).

      Attributes:
          child_frame_id: The frame whose pose is described by this odometry (e.g. ``"base_link"``).
          poses: Cached rosbags PoseStamped messages (rebuilt after any mutation).
          poses_rclpy: Cached rclpy/rospy PoseStamped messages (rebuilt after any mutation).
      """

      # Define odometry-specific data attributes
      child_frame_id: str
      poses: list  # Saved nav_msgs/msg/Pose for rosbags
      poses_rclpy: list  # Saved nav_msgs/msg/Pose for rclpy

      def __init__(self, frame_id: str, child_frame_id: str, timestamps: Union[np.ndarray, list],
                   positions: Union[np.ndarray, list], orientations: Union[np.ndarray, list], frame: CoordinateFrame):

          # Copy initial values into attributes
          super().__init__(frame_id, timestamps, positions, orientations, frame)
          self.child_frame_id: str = child_frame_id
          self.poses: list = []
          self.poses_rclpy: list = []

          # Check to ensure that all arrays have same length
          if len(self.timestamps) != len(self.positions) or len(self.positions) != len(self.orientations):
              raise ValueError("Lengths of timestamp, position, and orientation arrays are not equal!")
  ```


### Comprehensive Test Standard
All tests must meet the following bar — not just "does it run" but "does it catch bugs":

1. Full input coverage: Cover all meaningful inputs, including edge cases (empty, zero, one item, maximum, invalid). For infinite input spaces, cover a representative sample including boundaries.
2. Bug-detection strength: If a small, plausible implementation change were introduced (off-by-one, wrong condition, missing branch), at least one test must fail. Tests that only verify the happy path do not meet this bar.
3. Branch coverage: Every code branch (if/else, try/except, early return) must be exercised by at least one test case.

Apply this standard when writing new tests and when reviewing or extending existing ones. Use tools like coverage (examples found in .github/workflows/python_test.yml) to help ensure successful compliance with requirement #3.


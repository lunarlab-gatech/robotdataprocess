import getpass
from pathlib import Path
from robotdataprocess.data_types.LiDARData import LiDARData
from robotdataprocess.data_types.Data import CoordinateFrame

def main():
    # Enter desired configuration here
    dataset_num = "SmallTownSequence"
    user = getpass.getuser()
    input_dir = '/media/' + user + '/T73/Hercules_datasets/' + dataset_num + '/data'
    robot_name = "Husky1"
    num_channels = 16
    v_min_angle = -20
    v_max_angle = 20

    # Make directory paths
    input_path = Path(input_dir).absolute()

    # Extract LiDAR data
    lidar_data = LiDARData.from_npy_files(input_path / robot_name / "lidar", robot_name + '/lidar_link', CoordinateFrame.NED)
    lidar_data.calculate_point_channels(num_channels, v_min_angle, v_max_angle)
    lidar_data.make_dense()

    # Visualize it
    lidar_data.visualize(interval_ms=100, plot_lims=(-50.0, 50.0))

if __name__ == "__main__":
    main()

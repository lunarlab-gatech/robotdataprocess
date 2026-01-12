import getpass
from robotdataprocess.data_types.LiDARData import LiDARData

# Example usage of LiDARData class
if __name__ == "__main__":

    # Path to the folder containing .npy files
    user = getpass.getuser()
    npy_folder_path = '/media/' + user + '/hercules-collect/raw_data_hercules/ausenv_stereo3center_2ugvuav_calib_752x480/Drone1/lidar'
    frame_id = "lidar_frame"

    # Load LiDAR data from .npy files
    lidar_data = LiDARData.from_npy_files(npy_folder_path, frame_id)

    # Print some information about the loaded data
    print(f"Loaded LiDAR data with frame ID: {lidar_data.frame_id}")
    print(f"Number of LiDAR scans: {lidar_data.len()}")
    print(f"Timestamps: {lidar_data.timestamps}")
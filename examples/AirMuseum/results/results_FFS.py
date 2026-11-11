import getpass
from pathlib import Path
from robotdataprocess import ImageDataOnDisk

def main():
    dataset_seq = "Scenario5"
    robot_names = ["drone", "robotA", "robotB", "robotC"]
    max_depth = 30.0

    user = getpass.getuser()
    results_dir = Path('/media') / user / 'T73' / 'AirMuseum_dataset' / dataset_seq / 'results' / 'Fast-FoundationStereo'
    output_dir = Path('/home') / user / 'Research' / 'robotdataprocess' / 'figures' / 'airmuseum' / dataset_seq / 'videos_FFS'

    for robot_name in robot_names:
        depth_data = ImageDataOnDisk.from_npy_files(results_dir / robot_name / 'depth', f'{robot_name}/FoundationStereo')
        depth_data.depth_to_rgb(max_depth, reverse=True)
        depth_data.to_mp4(output_dir / f'{robot_name}.mp4', fps=20, video_duration_sec=20.0)

if __name__ == "__main__":
    main()

import getpass
from pathlib import Path
from robotdataprocess import ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData

def main():
    """
    Convert each robot's front camera images into an .mp4 video for the
    HERCULES dataset.

    See :meth:`ImageData.to_mp4` for the conversion itself.
    """
    dataset_seq = "V2.3.AP"
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]

    user = getpass.getuser()
    data_dir = Path('/media') / user / 'T73' / 'Hercules_datasets' / dataset_seq / 'data'
    output_dir = Path('/home/dbutterfield3/Research/robotdataprocess/figures/hercules') / dataset_seq / 'videos'

    for robot_name in robot_names:
        image_data = ImageDataOnDisk.from_image_files(data_dir / robot_name / 'rgb_stereo_left', 'front_center_Scene')
        image_data.to_mp4(output_dir / f'{robot_name}.mp4', fps=20, video_duration_sec=20.0)

if __name__ == "__main__":
    main()

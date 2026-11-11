from getpass import getuser
from pathlib import Path
import matplotlib.pyplot as plt
from robotdataprocess import ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData

def main():

    dataset_num: str = "SmallTownSequence"
    robot_name: str = "Drone1"
    frame_skip: int = 5
    input_path: Path = Path('/media') / getuser() / 'T73' / 'Hercules_datasets' / dataset_num / 'data' / robot_name
    image_data = ImageDataOnDisk.from_image_files(input_path / 'rgb_stereo_left', 'front_center_Scene')
    image_data.to_encoding(ImageData.ImageEncoding.RGB8)

    fig, ax = plt.subplots()
    im = ax.imshow(image_data.images[0], cmap="gray")
    ax.set_title(f"Frame 0/{len(image_data.images)}")

    for i in range(0, len(image_data.images), frame_skip):
        im.set_data(image_data.images[i])
        ax.set_title(f"Frame {i}/{len(image_data.images)}")
        if i == 0:
            plt.pause(5)
        plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    main()

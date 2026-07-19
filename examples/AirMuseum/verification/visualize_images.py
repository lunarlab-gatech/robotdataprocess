from getpass import getuser
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from robotdataprocess import ImageDataOnDisk
from robotdataprocess.data_types.ImageData.ImageData import ImageData

def main():

    scenario: str = "Scenario3"
    robot_name: str = "drone"
    bag_path: Path = Path('/media') / getuser() / 'T73' / 'AirMuseum_dataset' / scenario / 'data' / robot_name / 'cam100_imu.bag'
    image_data = ImageDataOnDisk.from_ros1_bag(bag_path, f'/{robot_name}/cam100/image_raw')

    fig, ax = plt.subplots()
    im = ax.imshow(_to_rgb(image_data, 0), cmap="gray")
    ax.set_title(f"Frame 0/{len(image_data.images)}")

    for i in range(len(image_data.images)):
        im.set_data(_to_rgb(image_data, i))
        ax.set_title(f"Frame {i}/{len(image_data.images)}")
        plt.pause(0.001)

    plt.show()

def _to_rgb(image_data: ImageDataOnDisk, i: int):
    image = image_data.images[i]
    if image_data.encoding == ImageData.ImageEncoding.BGR8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":
    main()
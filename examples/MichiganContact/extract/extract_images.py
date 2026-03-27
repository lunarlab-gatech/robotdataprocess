
from robotdataprocess import ImageDataOnDisk, ImageDataInMemory

image_data = ImageDataInMemory.from_ros2_bag('/media/dbutterfield3/T73/MichiganContact_dataset/V1.0/data/2021-06-05-17-12-39', '/camera/color/image_raw', '/media/dbutterfield3/T73/MichiganContact_dataset/V1.0/data/2021-06-05-17-12-39/temp_images')
image_data.to_image_files('/media/dbutterfield3/T73/MichiganContact_dataset/V1.0/extract/2021-06-05-17-12-39/images')
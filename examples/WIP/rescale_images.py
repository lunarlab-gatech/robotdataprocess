from robotdataprocess.data_types.ImageData import ImageData

def main():
    images = ImageData.from_npy('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/left_rectified')
    images.downscale_by_factor(4)
    images.to_npy('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/left_rectified_small')

if __name__ == "__main__":
    main()
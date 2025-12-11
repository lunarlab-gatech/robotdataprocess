from robotdataprocess import ImageData

def main():
    # Load Left rectified from GRaCo Dataset
    left_img_rectified = ImageData.from_image_files("/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/left_rectified", "camera_21239776")

    # Save it as a single .npy file for use with ROMAN
    left_img_rectified.to_npy("/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/left_rectified")

if __name__ == "__main__":
    main()
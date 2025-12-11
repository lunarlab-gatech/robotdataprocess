from robotdataprocess import ImageData

def main():
    # Load Estimated Depth from FoundationStereo (of GRaCo Dataset)
    depth_data = ImageData.from_npy_files("/home/dbutterfield3/Research/FoundationStereo/outputs/GRaCo/ground_05/depth", "camera_21239776")

    # Save it as a single .npy file for use with ROMAN
    depth_data.to_npy("/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/depth")

if __name__ == "__main__":
    main()
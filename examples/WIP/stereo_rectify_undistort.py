import numpy as np
from robotdataprocess import ImageData, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper

def main():
    # Extract stereo imagery from GRaCo dataset
    left_data = ImageData.from_ros2_bag('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05', '/camera_left/image_raw', '/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/left')
    right_data = ImageData.from_ros2_bag('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05', '/camera_right/image_raw', '/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/npy/right')
    # left_data = ImageData.from_npy('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-04-extract/npy/left')
    # right_data = ImageData.from_npy('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-04-extract/npy/right')

    # Define camera intrinsics
    K1 = np.array([[940.862825677534,                0, 799.1626975233576],
                   [               0, 938.554923506332,  559.295406893583],
                   [               0,                0,                 0]], dtype=np.float64)
    D1 = np.array([[-0.1008504099655989, 0.08905706623788286, 0.0007516966627205781,-0.0011958374307601393]], dtype=np.float64)

    K2 = np.array([[934.5190744321391,                0, 792.8073165035943],
                   [                0, 932.525429508503, 562.7061769000949],
                   [                0,                0,                 0]], dtype=np.float64)
    D2 = np.array([[-0.10093604150569942, 0.08966460684307566, 0.0006752623328139636,-0.0015031965111152103]], dtype=np.float64)

    # Define camera extrinsics
    R = np.array([[0.99965907,  0.01921468, -0.01767879],
                  [-0.01922979,  0.99981486, -0.00068469],
                  [0.01766236,  0.00102441,  0.99984348]], dtype=np.float64)
    T = np.array([-0.23223044, 0.00288665, -0.000523], dtype=np.float64)

    # Rectify and Undistort them
    left_rectified, right_rectified, K1_new, K2_new = left_data.stereo_undistort_and_rectify(right_data, K1, D1, K2, D2, R, T)

    # Save as raw image files
    left_rectified.to_image_files("/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/left_rectified")
    right_rectified.to_image_files("/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/right_rectified")

if __name__ == "__main__":
    main()
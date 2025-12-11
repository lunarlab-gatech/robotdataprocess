from robotdataprocess.data_types.OdometryData import OdometryData

def main():
    gt_pose = OdometryData.from_ros2_bag('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05', '/gnss/ground_truth')
    gt_pose.to_csv('/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-05-extract/gt_pose.csv')
    gt_pose.visualize([], ['GT Odometry (GRaCo Ground-05)'])

if __name__ == "__main__":
    main()
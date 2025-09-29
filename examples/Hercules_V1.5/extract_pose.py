from robotdataprocess import OdometryData, CoordinateFrame

def main():
    robot_name = "Drone2"

    # Load the odometry data
    odom_data = OdometryData.from_txt_file('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/data/' + robot_name + '/pose_world_frame.txt', robot_name + '/odom', robot_name + '/ground_truth/base_link', CoordinateFrame.NED)

    # Convert to the FLU coordinate frame
    odom_data.to_FLU_frame()

    # Save back to a csv file
    odom_data.to_csv('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/extract/files_for_roman_baseline/' + robot_name + '/poseGT.csv')

if __name__ == "__main__":
    main()
from robotdataprocess import ImageData, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper

def main():
    robot_name = "Drone2"

    # Extract RGB and IMU from Hercules v1.5
    imu_data = ImuData.from_txt_file('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/data/' + robot_name + '/synthetic_imu.txt', 
                                     '' + robot_name + '/base_link', CoordinateFrame.NED)
    pose_data = OdometryData.from_txt_file('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/data/' + robot_name + '/pose_world_frame.txt', 'world', 'body', CoordinateFrame.NED)
    image_data = ImageData.from_image_files('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/data/' + robot_name + '/rgb', '' + robot_name + '/front_center_Scene')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    pose_data.to_FLU_frame()
    pose_data.shift_to_start_at_identity()

    # Leave the IMU data in the NED frame (I believe that VINS-Mono actually adjusts internally)
    imu_data.frame = CoordinateFrame.FLU # This lets us write it into a ROS bag without an error, without actually changing data

    # Save it into a ROS2 Humble bag
    Ros2BagWrapper.write_data_to_rosbag('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/extract/bags_for_vins_mono/' + robot_name,
             [imu_data, image_data,  pose_data,       pose_data],
             [  '/imu',    '/cam0', '/odom_gt', '/odom_gt/path'],
             [    None,       None, "Odometry",          "Path"], 
             None)

if __name__ == "__main__":
    main()
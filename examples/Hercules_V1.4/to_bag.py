from robotdataprocess import ImageData, ImuData, OdometryData, CoordinateFrame
from robotdataprocess.rosbag.Ros2BagWrapper import Ros2BagWrapper

def main():
    robot_name = "Drone1"

    # Extract RGB and IMU from Hercules v1.3
    imu_data = ImuData.from_txt_file('/mnt/d/Hercules/V1.4.1/data/' + robot_name + '/partial_imu.txt', 
                                     '' + robot_name + '/base_link', CoordinateFrame.NED)
    odom_data = OdometryData.from_txt_file('/mnt/d/Hercules/V1.4.1/data/' + robot_name + '/partial_odom.txt', 'world', 'body', CoordinateFrame.NED)
    image_data = ImageData.from_image_files('/mnt/d/Hercules/V1.4.1/data/' + robot_name + '/partial_rgb', '' + robot_name + '/front_center_scene')
    depth_data = ImageData.from_npy_files('/mnt/d/Hercules/V1.4.1/data/' + robot_name + '/partial_depth', '' + robot_name + '/depth')

    # Convert data from NED frame to ROS frame (and make sure it is at the identity)
    odom_data.to_FLU_frame()
    odom_data.shift_to_start_at_identity()

    # Leave the IMU data in the NED frame (I believe that VINS-Mono actually adjusts internally)
    imu_data.frame = CoordinateFrame.FLU # This lets us write it into a ROS bag without an error, without actually changing data

    # Save it into a ROS2 Humble bag
    Ros2BagWrapper.write_data_to_rosbag('/mnt/d/Hercules/V1.4.1/data/bags/' + 'partial_drone1.bag',
             [imu_data, image_data,  odom_data,       odom_data, depth_data], 
             [  '/imu',    '/cam0', '/odom_gt', '/odom_gt/path', '/depth'], 
             [    None,       None, "Odometry",          "Path", None], 
             None)

if __name__ == "__main__":
    main()
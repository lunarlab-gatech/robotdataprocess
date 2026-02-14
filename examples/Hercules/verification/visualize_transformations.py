import getpass
from robotdataprocess import TransformationData, CoordinateFrame

def main():
    dataset_version = "V2.3.AC"
    user = getpass.getuser()
    settings_path = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version + "/data/settings.json"
    robot_name = "Husky1"
    sensor_type = "camera"
    sensor_names = ["stereo_left", "stereo_right"]

    trans = []
    for sensor_name in sensor_names:
        H_R_to_C = TransformationData.from_HERCULES_settings_json(settings_path, robot_name, sensor_type, sensor_name)
        H_C_to_O = TransformationData.optical_wrt_camera(CoordinateFrame.NED, frame_id=sensor_name)
        H_R_to_O = H_R_to_C.apply_transformation_right_side(H_C_to_O)
        trans.append(H_R_to_O)
        H_R_to_O_FLU = H_R_to_O.to_coordinate_frame(CoordinateFrame.FLU)
        trans.append(H_R_to_O_FLU)
    
    TransformationData.visualize(trans, axes_length=0.5)

if __name__ == "__main__":
    main()
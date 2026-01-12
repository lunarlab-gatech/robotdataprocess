import getpass
from robotdataprocess.data_types.OdometryData import OdometryData
from robotdataprocess import CoordinateFrame

def main():
    robot_names = ["Drone2"]
    dataset_name = "V1.4.1"
    user = getpass.getuser()
    file_path = '/home/' + user + '/Desktop/data/Hercules_datasets/'+dataset_name+'/extract/files_for_roman_baseline/'

    # Load csv files
    data: list[OdometryData] = []
    for name in robot_names:
        d = OdometryData.from_csv(file_path + name +'/poseGT.csv', "odom", 'ground_truth/base_link', CoordinateFrame.FLU, True, None)
        data.append(d)

    # Load VINS estimated results
    for name in robot_names:
        d = OdometryData.from_csv(file_path + name +'/vins_result_loop_reformatted.csv', "odom", 'base_link', CoordinateFrame.FLU, True, None)
        data.append(d)
        d = OdometryData.from_csv(file_path + name +'/vins_result_no_loop_reformatted.csv', "odom", 'base_link', CoordinateFrame.FLU, True, None)
        data.append(d)

    data[0].shift_to_start_at_identity()

    # Visualize it
    robot_names_alt = [robot_names[0] + "(VINS-Mono Loop)", robot_names[0] +"(VINS-Mono No Loop)"]
    print(len(data[1:]))
    data[0].visualize(data[1:], robot_names + robot_names_alt)

if __name__ == "__main__":
    main()
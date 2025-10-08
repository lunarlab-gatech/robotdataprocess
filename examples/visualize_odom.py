from robotdataprocess.data_types.OdometryData import OdometryData
from robotdataprocess import CoordinateFrame

def main():
    robot_names = ["Drone2"]

    # Load csv files
    data: list[OdometryData] = []
    for name in robot_names:
        d = OdometryData.from_csv('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/extract/files_for_roman_baseline/'+name+'/poseGT.csv', "odom", 'ground_truth/base_link', CoordinateFrame.FLU, True, None)
        data.append(d)

    # Load VINS estimated results
    for name in robot_names:
        d = OdometryData.from_csv('/home/dbutterfield3/Desktop/data/Hercules_datasets/V1.5/extract/files_for_roman_baseline/'+name+'/vins_result_no_loop_reformatted.csv', "odom", 'base_link', CoordinateFrame.FLU, True, None)
        data.append(d)

    data[0].shift_to_start_at_identity()

    # Visualize it
    robot_names_alt = [x + "(VINS-Mono)" for x in robot_names]
    data[0].visualize(data[1:], robot_names + robot_names_alt)

if __name__ == "__main__":
    main()
import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData, CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():
    # Load all GT and plot together
    robot_names = ["Husky1", "Husky2", "Drone1", "Drone2"]
    dataset_version = "V2.3.AC"

    gt_data_lst = []

    for robot_name in robot_names:
        print("Processing results for robot:", robot_name)
        user = getpass.getuser()
        dataset_folder = '/media/' + user + '/T73/Hercules_datasets/' + dataset_version
                    
        gt_data = OdometryData.from_csv(dataset_folder + "/extract/files_for_roman_baseline/" + robot_name + '/poseGT.csv', 
                                        "world", "robot", CoordinateFrame.FLU, True, None)
        gt_data_lst.append(gt_data)
    
    # Visualize all ground truth together
    gt_data_lst[0].visualize(gt_data_lst[1:], [robot_names[i] + " GT" for i in range(0, len(robot_names))], 5, 400)

if __name__ == "__main__":
    main()
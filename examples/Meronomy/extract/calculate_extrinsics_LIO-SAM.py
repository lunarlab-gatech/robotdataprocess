import getpass
import numpy as np
from robotdataprocess import TransformationData

def main():
    user = getpass.getuser()
    dataset_num = "V1.2"
    input_dir = '/media/' + user + '/T73/Meronomy_datasets/' + dataset_num + '/data'
    robot_name = "Drone1"

    # Load the I->L Transformation
    trans = TransformationData.from_HERCULES_settings_json(input_dir + '/settings.json', robot_name, "Sensor", "LidarSensor1")

    # Calculate the L->I Transformation
    T = trans.invert().as_matrix()
    rot = T[:3, :3]
    trans_vec = T[:3, 3]

    print(T)

    # Print translation & rotation in the format LIO-SAM expects
    print(f"  extrinsicTrans:  [ {trans_vec[0]: .8f}, {trans_vec[1]: .8f}, {trans_vec[2]: .8f} ]")
    r = rot.flatten()
    print("  extrinsicRot:    [ "
        f"{r[0]: .8f}, {r[1]: .8f}, {r[2]: .8f},\n"
        "                     "
        f"{r[3]: .8f}, {r[4]: .8f}, {r[5]: .8f},\n"
        "                     "
        f"{r[6]: .8f}, {r[7]: .8f}, {r[8]: .8f} ]")
    print("  extrinsicRPY:    [ "
        f"{r[0]: .8f}, {r[1]: .8f}, {r[2]: .8f},\n"
        "                     "
        f"{r[3]: .8f}, {r[4]: .8f}, {r[5]: .8f},\n"
        "                     "
        f"{r[6]: .8f}, {r[7]: .8f}, {r[8]: .8f} ]")

if __name__ == "__main__":
    main()
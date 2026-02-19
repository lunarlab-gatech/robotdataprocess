from decimal import Decimal
import getpass
import numpy as np
from robotdataprocess.data_types.OdometryData import OdometryData
from robotdataprocess import CoordinateFrame
from scipy.spatial.transform import Rotation as R

def main():

    user = getpass.getuser()
    maplab = OdometryData.from_csv('/home/' + user + '/Desktop/data/vertex_poses_velocities_biases.csv', 
            "world", "robot", CoordinateFrame.ENU, True, [0, 3, 4, 5, 6, 7, 8, 9])
    vicon = OdometryData.from_csv('/home/' + user + '/Desktop/data/vicon_trajectory.csv', 
            "odom", 'base_link', CoordinateFrame.FLU, True, [0, 1, 2, 3, 7, 4, 5, 6])
    vicon.shift_to_start_at_identity()

    vicon_start_ts = 1403715271.70518
    maplab_start_ts = 1.40371527341214E+018
    vicon.crop_data(Decimal(vicon_start_ts), Decimal(vicon_start_ts + 40))
    maplab.crop_data(Decimal(maplab_start_ts), Decimal(maplab_start_ts + 4*10E9))

    # Updates the positions to be in the proper frame while preserving rotation information
    # Achieved via change of basis flipping X and Y
    R_FIRST = np.array([[-1, 0,  0],
                      [0, -1,  0],
                      [0,  0, 1]])
    maplab._convert_frame(R_FIRST)

    # Update the rotations to rotate along the same axes as ground truth
    R_NEXT = np.array([[0, 0,  1],
                       [0, 1,  0],
                       [-1, 0, 0]])
    maplab._ori_change_of_basis(R.from_matrix(R_NEXT))

    # Estimate R_result for now (SHOULD USE GT Robot->Camera transformation from dataset in proper frame (FLU here))
    R_maplab = R.from_quat(maplab.orientations[0])
    R_vicon = R.from_quat(vicon.orientations[0])
    R_result: R = R_vicon * R_maplab.inv()
    print(R_result.as_euler('xyz', degrees=True))

    # Rotate the axes to be pointed in the same directions as our GT
    maplab._ori_apply_rotation(R_result)

    # Visualizes the positions and orientations
    maplab.visualize_3D([vicon], ['Maplab', 'VICON'])

if __name__ == "__main__":
    main()
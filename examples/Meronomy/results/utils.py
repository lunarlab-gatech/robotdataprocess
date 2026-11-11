from dataclasses import dataclass, field
import getpass
from robotdataprocess import OdometryData, PathData
from typing import List, Optional

@dataclass
class LoadDataResult:
    est_data_robot0: Optional[OdometryData] = None
    est_data_robot1: Optional[OdometryData] = None
    est_data_lst: List[OdometryData] = field(default_factory=list)

    gt_data_robot0: Optional[OdometryData] = None
    gt_data_robot1: Optional[OdometryData] = None
    gt_data_lst: List[OdometryData] = field(default_factory=list)

def print_errors(metrics_dictionary: dict):
    """ Helper function to print desired metrics. """

    print("RMS ATE: ", metrics_dictionary.APE.translation_part.rmse)
    print("RMS RTE: ", metrics_dictionary.RPE.translation_part.rmse)
    print("Standard Deviation RTE: ", metrics_dictionary.RPE.translation_part.std)

    print("RMS APE Rotation Angle (Deg): ", metrics_dictionary.APE.rotation_angle_deg.rmse)
    print("RMS RTE Rotation Angle (Deg): ", metrics_dictionary.RPE.rotation_angle_deg.rmse)
    print("Standard Deviation RTE Rotation Angle (Rad): ", metrics_dictionary.RPE.rotation_angle_rad.std)

def plot_GT_vs_est_on_image(data: LoadDataResult, est_data_align: OdometryData, gt_data_align: OdometryData,
                                   dataset_name: str, robot0_name: str, robot1_name: str):

    # Seperate the aligned trajectories into their single-robot forms
    gt_data_align_list = PathData.seperate_PathData(data.gt_data_lst, gt_data_align)
    gt_data_align_robot0 = gt_data_align_list[0]
    gt_data_align_robot1 = gt_data_align_list[1]

    est_data_align_list = PathData.seperate_PathData(data.est_data_lst, est_data_align)
    est_data_align_robot0 = est_data_align_list[0]
    est_data_align_robot1 = est_data_align_list[1]

    # Get environment image path
    user = getpass.getuser()
    image_path = '/media/' + user + '/T73/GrAco_dataset/' + dataset_name + '/data/environment.png'
    x_edge = 691.216296

    # Define the mapping from robot name to color and robot_name to new name
    robot_name_to_color: dict = {
        "Husky1": "#D61AD0",
        "ground-06": "#12EF49",
        "Drone1": "#1A46D6",
        "aerial-08": "#E8EF12",
    }

    # Plot the results in 2D
    dataList =  [est_data_align_robot0, gt_data_align_robot0,  est_data_align_robot1,  gt_data_align_robot1]
    isGTList =  [                False,                 True,                  False,                  True]
    nameList =  [robot0_name, robot0_name, robot1_name, robot1_name]
    colorList = [robot_name_to_color[name] for name in nameList]
    PathData.visualize_2D(dataList, isGTList, colorList, nameList, no_background=True, line_width=3.0, show_grid=False, 
                       background_image_path=image_path, background_image_x_edge=x_edge, gt_color_lightness_range_val=12,
                       background_image_extent_offsets=(55, 80), no_border=True,
                       save_path='/home/dbutterfield3/Research/robotdataprocess/fig.pdf')
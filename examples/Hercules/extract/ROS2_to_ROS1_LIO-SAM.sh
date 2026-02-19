#!/bin/bash
# Converts ROS2 bags created by extract_data_LIO-SAM.py to ROS1 bags using rosbags-convert

# Get the current user
USER=$(whoami)

# Define paths
DATASET_NUM="V2.4.F"
BASE_DIR="/media/${USER}/T73/Hercules_datasets/${DATASET_NUM}/extract/bags_for_LIO-SAM"

# Robot names (must match extract_data_LIO-SAM.py)
ROBOTS=("Husky1" "Husky2" "Drone1" "Drone2")

# Convert each robot's ROS2 bag to ROS1
for ROBOT in "${ROBOTS[@]}"; do
    INPUT_BAG="${BASE_DIR}/${ROBOT}/"
    OUTPUT_BAG="${BASE_DIR}/${ROBOT}.bag"

    if [ -d "${INPUT_BAG}" ]; then
        echo "Converting ${ROBOT} ROS2 bag to ROS1..."
        rosbags-convert "${INPUT_BAG}" --dst "${OUTPUT_BAG}"
        echo "Done: ${OUTPUT_BAG}"
    else
        echo "Warning: ${INPUT_BAG} does not exist, skipping..."
    fi
done

echo "All conversions complete."

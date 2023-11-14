#!/bin/bash

# CMD: 
# bash p2p_multiple_evaluations.sh /mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/p2p/online /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt
# bash p2p_multiple_evaluations.sh /mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/p2p/offline /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./p2p_run_multiple_evaluations.sh main_directory ref_poses_path"
    exit 1
fi

# Extract arguments
main_directory="$1"
ref_poses_path="$2"

# The path to your previously created script
script_path="./evo.sh"

# Loop over all directories in the main_directory
for dir in "$main_directory"/*/; do
    # Construct the estimated_poses_path
    estimated_poses_path="${dir}estimated_tf_kitti.txt"

    # Call the previously created script with the current estimated_poses_path
    if [ -f "$estimated_poses_path" ]; then
        bash "$script_path" "$ref_poses_path" "$estimated_poses_path"
    else
        echo "File not found in $dir, skipping this directory."
    fi
done

echo "All evaluations complete."
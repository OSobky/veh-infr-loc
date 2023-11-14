#!/bin/bash

# CMD:
# bash kiss_multiple_evaluations.sh /mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/kiss-icp /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./kiss_run_multiple_evaluations.sh main_directory ref_poses_path"
    exit 1
fi

# Extract arguments
main_directory="$1"
ref_poses_path="$2"

# The path to your previously created script
script_path="./evo.sh"

# Loop through offline and online directories
for mode in offline online; do
    # Loop through ground and no_ground directories
    for type in ground no_ground; do
        # Determine the filename based on the type
        if [ "$type" == "ground" ]; then
            filename="matched_poses_kitti.txt"
        else
            filename="ground_removed_pcds_0.0_poses_kitti.txt"
        fi

        # Loop over all directories inside ground/no_ground
        for subdir in "$main_directory/$mode/$type"/*/; do
            # Skip the 'latest' folder
            if [[ "$subdir" != *"/latest/"* ]]; then
                estimated_poses_path="${subdir}${filename}"

                # Call the previously created script with the current estimated_poses_path
                if [ -f "$estimated_poses_path" ]; then
                    bash "$script_path" "$ref_poses_path" "$estimated_poses_path"
                else
                    echo "File not found in $subdir, skipping this directory."
                fi
            fi
        done
    done
done

echo "All evaluations complete."
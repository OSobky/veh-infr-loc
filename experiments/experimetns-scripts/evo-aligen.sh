#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_evo_evaluation.sh ref_poses_path estimated_poses_path"
    exit 1
fi

# Extract arguments
ref_poses_path="$1"
estimated_poses_path="$2"

# Deduce parent directory of estimated_poses_path for saving plots
parent_dir=$(dirname "$estimated_poses_path")

# Values for the -r option
declare -a r_values=("full" "trans_part" "rot_part")

# Loop through each value for -r and run evo_ape and evo_rpe commands, saving logs each time
for r_val in "${r_values[@]}"
do
    # Run evo_ape command and save logs
    evo_ape kitti "$ref_poses_path" "$estimated_poses_path" --no_warnings -a -r "$r_val" -v --plot_mode xy --save_results "$parent_dir/absolute_pose_error_${r_val}.zip" --save_plot "$parent_dir/ape_plot_${r_val}.png" > "$parent_dir/ape_log_${r_val}.txt" 2>&1

    # Run evo_rpe command and save logs
    evo_rpe kitti "$ref_poses_path" "$estimated_poses_path" --no_warnings -a -r "$r_val" -v --plot_mode xy --save_results "$parent_dir/relative_pose_error_${r_val}.zip" --save_plot "$parent_dir/rpe_plot_${r_val}.png" > "$parent_dir/rpe_log_${r_val}.txt" 2>&1
done

# Run evo_traj command and save logs
evo_traj kitti "$estimated_poses_path" --ref="$ref_poses_path" --no_warnings -a --plot_mode=xy --save_plot "$parent_dir/traj_plot.png" > "$parent_dir/traj_log.txt" 2>&1

echo "Evaluation complete. Plots and logs saved to $parent_dir"
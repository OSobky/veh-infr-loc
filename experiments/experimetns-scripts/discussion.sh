#!/bin/bash

# Default parent directories containing result .zip files
PARENT_DIRS=(
  "/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/kiss-icp/offline/ground/2023-10-06_19-34-48"
  "/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/kiss-icp/offline/no_ground/2023-10-06_20-51-25"
  # ... Add as many directories as needed
)

OUTPUT_DIR="/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/discussion/test"

# Function to run evo_res for a given mode (ape/rpe) and error type (full/trans_part/rot_part)
run_evo_res() {
    local mode="$1"
    local error_type="$2"
    local files=()

    # Collect all the files for the given mode and error type across all parent directories
    for dir in "${PARENT_DIRS[@]}"; do
        files+=("${dir}/${mode}_pose_error_${error_type}.zip")
    done

    # Run evo_res
    evo_res --no_warnings --use_filenames  --save_plot "${OUTPUT_DIR}/${mode}_${error_type}.png" "${files[@]}" > "${OUTPUT_DIR}/${mode}_${error_type}_log.txt" 2>&1
}

# Run for all combinations
for mode in absolute relative; do
    for error_type in full trans_part rot_part; do
        run_evo_res "$mode" "$error_type"
    done
done

echo "All evaluations complete."
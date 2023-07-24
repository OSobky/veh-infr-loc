#!/bin/bash

# Example:
#bash src/bag_extractor/scripts/convert_pcd_binary_to_ascii.sh <INPUT_FOLDER_PATH_PCD_BINARY> <OUTPUT_FOLDER_PATH_PCD_ASCII>

# NOTE: to use pcl_convert_pcd_ascii_binary you first need to install the pcl-tools package:
# sudo apt install pcl-tools

INPUT_FOLDER_PATH_PCD_BINARY=$1
OUTPUT_FOLDER_PATH_PCD_ASCII=$2

for file_path in ${INPUT_FOLDER_PATH_PCD_BINARY}/*.pcd; do
  file_name="${file_path##*/}"
  pcl_convert_pcd_ascii_binary ${file_path} ${OUTPUT_FOLDER_PATH_PCD_ASCII}/${file_name} 0
done

import open3d as o3d
import os
import numpy as np
import argparse

def pcd_to_bin(input_folder):
    # Create the output folder in the parent directory
    parent_dir = os.path.dirname(input_folder)
    output_folder = os.path.join(parent_dir, "bin_files")
    os.makedirs(output_folder, exist_ok=True)

    # Loop over all .pcd files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pcd"):
            # Load the point cloud
            pcd_path = os.path.join(input_folder, file_name)
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            # Filter out invalid points (e.g., -0, 0, -0)
            mask = np.all(points != 0, axis=1)
            valid_points = points[mask]

            # Convert to float32 and save as .bin file
            bin_data = valid_points.astype(np.float32).tobytes()
            bin_path = os.path.join(output_folder, file_name.replace(".pcd", ".bin"))
            with open(bin_path, "wb") as bin_file:
                bin_file.write(bin_data)

            print(f"Converted {pcd_path} to {bin_path}")

def main():
    parser = argparse.ArgumentParser(description="Converts .pcd files to .bin format.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing .pcd files.")
    args = parser.parse_args()
    pcd_to_bin(args.input_folder)

if __name__ == "__main__":
    main()


# Example usage:
# python open3d_registeration/pcd_to_bin.py <input_folder>
# python open3d_registeration/pcd_to_bin.py /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/matched
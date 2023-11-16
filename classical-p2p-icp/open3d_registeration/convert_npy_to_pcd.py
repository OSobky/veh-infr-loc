import numpy as np
import open3d as o3d
import os
import argparse

def npy_to_pcd(input_path: str, visualize: bool = False) -> None:
    # Load the .npy file
    point_cloud_data = np.load(input_path)
   
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Save the point cloud in .pcd format
    parent_dir = os.path.dirname(input_path)
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(parent_dir, f"{file_name}.pcd")

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved to {output_path}")

    # Visualize the point cloud if visualize flag is set
    if visualize:
        pcd = o3d.io.read_point_cloud(output_path)
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", width=800, height=600)

def main():
    parser = argparse.ArgumentParser(description="Convert .npy point clouds to .pcd format using Open3D.")
    parser.add_argument("input_path", type=str, help="Path to the .npy file to be converted.")
    parser.add_argument("--viz", action="store_true", help="Visualize the created .pcd file after conversion.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_path) or not args.input_path.endswith('.npy'):
        print("Please provide a valid path to a .npy file.")
        exit(1)

    npy_to_pcd(args.input_path, visualize=args.viz)

if __name__ == "__main__":
    main()
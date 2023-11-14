import open3d as o3d
import numpy as np
import os
import argparse

def remove_ground(point_cloud, ground_height=0):
    column = np.asarray(point_cloud.points)[:,2]
    filtered_frame = np.asarray(point_cloud.points)[column > ground_height]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_frame)
    return pcd

def filter_origin_points(pcd, threshold=1e-6):
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    
    # Create a mask for points that are close to the origin
    mask = np.linalg.norm(points, axis=1) > threshold
    
    # Apply the mask to filter out points close to the origin
    filtered_points = points[mask]
    
    # Create a new point cloud with the filtered points
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    return filtered_pcd

def downsample_pointcloud(pcd, voxel_size=0.05):
    
    # Downsample the point cloud
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return downsampled_pcd

def process_pcds_in_folder(pcd_folder, ground_height=0, remove_floor=True,remove_origin_points=True, downsample=True):
    parent_folder = os.path.dirname(pcd_folder)
    output_folder_str = ""

    if remove_floor:
        output_folder_str = output_folder_str + f"__{ground_height}"
    if remove_origin_points:
        output_folder_str = output_folder_str +  "_no_origin"
    if downsample:
        output_folder_str = output_folder_str + f"_downsampled"
    

    output_folder = os.path.join(parent_folder, output_folder_str)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(pcd_folder):
        if file.endswith('.pcd'):
            print(f"Processing {file}...")
            pcd_path = os.path.join(pcd_folder, file)
            point_cloud = o3d.io.read_point_cloud(pcd_path, remove_nan_points=True, remove_infinite_points=True)
            
            if remove_floor:
                point_cloud = remove_ground(point_cloud, ground_height)
            if remove_origin_points:
                point_cloud = filter_origin_points(point_cloud)
            if downsample:
                point_cloud = downsample_pointcloud(point_cloud)


            output_path = os.path.join(output_folder, file)
            
            o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True)
            print(f"Saved ground-removed point cloud to {output_path}")

def main(args):
    if not os.path.exists(args.pcd_folder):
        print(f"Error: The folder '{args.pcd_folder}' does not exist.")
        return

    process_pcds_in_folder(args.pcd_folder, args.ground_height, remove_floor=False, remove_origin_points=True, downsample=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove ground from point clouds in the specified folder.')
    parser.add_argument('pcd_folder', type=str, help='Path to the folder containing the PCD files')
    parser.add_argument('ground_height', type=float, help='Height threshold for ground removal')
    
    args = parser.parse_args()
    main(args)

    # Example usage:
    # python open3d_registeration/remove_ground_save.py /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/matched -7.25
    # python open3d_registeration/remove_ground_save.py /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched 0
import open3d as o3d
import numpy as np
import os

from utils import initial_tf_from_gps

def transform_pcd(src_dir, gps_dir, imu_dir):

    transformed_dir = os.path.join(os.path.dirname(src_dir), 'transformed_cleaned_dir')
    if not os.path.exists(transformed_dir):
        os.makedirs(transformed_dir)
    for i, (src_pcd_filename, gps_filename, imu_filename) in enumerate(zip(
        sorted(os.listdir(src_dir)),sorted(os.listdir(gps_dir)), sorted(os.listdir(imu_dir)), 
        )):
        
        pcd_path = os.path.join(src_dir,src_pcd_filename)
        pcd = o3d.io.read_point_cloud(pcd_path, remove_nan_points=True, remove_infinite_points=True)
        transformation = initial_tf_from_gps(os.path.join(gps_dir,gps_filename), 
                                         os.path.join(imu_dir,imu_filename))

        pcd.transform(transformation)
        o3d.io.write_point_cloud(os.path.join(transformed_dir,src_pcd_filename), pcd, write_ascii = True)

        # # reread the transformed point cloud and visualize it
        # pcd_load = o3d.io.read_point_cloud(os.path.join(transformed_dir,src_pcd_filename))
        # o3d.visualization.draw_geometries([pcd_load], )
        

def main():
    src_dir = '/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched'
    gps_dir = '/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/03_gps/04_gps_position_drive/json/matched'
    imu_dir = '/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/04_imu/04_imu_rotations_drive/json/matched'
    transform_pcd(src_dir, gps_dir, imu_dir)

if __name__ == "__main__":
    main()
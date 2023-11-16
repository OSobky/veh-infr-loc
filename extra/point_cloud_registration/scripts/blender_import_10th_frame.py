import math
import os

import bpy

if __name__ == "__main__":


    folder_path_point_cloud_source = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/transformed/"
    folder_path_point_cloud_target = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/transformed/"
    output_dir = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/Ground_truth_export_Blend/"
    count = 0
    for point_cloud_source_filename, point_cloud_target_filename,  in zip(
                sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target))):
             
        if count % 10 ==1:
            
            bpy.data.collections.new(name  = point_cloud_source_filename.split("_")[0])
            bpy.context.scene.collection.children.link(bpy.data.collections[point_cloud_source_filename.split("_")[0]])
            bpy.context.view_layer.active_layer_collection = bpy.data.collections[point_cloud_source_filename.split("_")[0]]
            
            bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_source + point_cloud_source_filename)
            bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_target + point_cloud_target_filename)
        
       
        
        count+=1
        
        
        

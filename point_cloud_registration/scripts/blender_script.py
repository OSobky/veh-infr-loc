import math
import os

import bpy

if __name__ == "__main__":


    folder_path_point_cloud_source = "/hom  e/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/transformed/"
    folder_path_point_cloud_target = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/transformed/"
    output_dir = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/Ground_truth_export_Blend/"
    for point_cloud_source_filename, point_cloud_target_filename,  in zip(
                sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target))):
        
        
        bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_source + point_cloud_source_filename)
        bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_target + point_cloud_target_filename)
        
        pcd_source = bpy.context.scene.objects[point_cloud_source_filename.split(".")[0]]
        pcd_target = bpy.context.scene.objects[point_cloud_target_filename.split(".")[0]]
        
        pcd_source.location.x += -0.58
        pcd_source.location.y += 0.91
        pcd_source.location.z += 0.1
        
        pcd_source.rotation_euler.x += -0.6 * math.pi/180
        pcd_source.rotation_euler.y += -0.4 * math.pi/180
        pcd_source.rotation_euler.z += -0.9 * math.pi/180
        
                
        bpy.ops.export_mesh.pcd(filepath= output_dir + point_cloud_source_filename.split("_")[0] + "_" + point_cloud_source_filename.split("_")[1] + "_ground_truth.pcd" )
        
        bpy.ops.object.delete() 
        
        
        
        break

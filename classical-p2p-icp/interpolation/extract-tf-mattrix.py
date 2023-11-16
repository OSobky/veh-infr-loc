import math
import os
from os.path import dirname, join

import bpy
import numpy as np

if __name__ == "__main__":


#    folder_path_point_cloud_source = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/transformed/"
#    folder_path_point_cloud_target = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/transformed/"
#    output_dir = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/Ground_truth_export_Blend/"
#    count = 0
#    for point_cloud_source_filename, point_cloud_target_filename,  in zip(
#                sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target))):
#             
#        if count % 10 ==1:
#            
#            bpy.data.collections.new(name  = point_cloud_source_filename.split("_")[0])
#            bpy.context.scene.collection.children.link(bpy.data.collections[point_cloud_source_filename.split("_")[0]])
#            bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[point_cloud_source_filename.split("_")[0]]
#            
#            
#            bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_source + point_cloud_source_filename)
#            bpy.ops.import_mesh.pcd(filepath=folder_path_point_cloud_target + point_cloud_target_filename)
#        
#       
#        
#        count+=1
#        
                
        
    if bpy.context.object and bpy.data.filepath:
      
#        fname = bpy.context.object.name + ".matrix"
#        fpath = join( dirname( bpy.data.filepath ),  fname )

#        f = open( fpath, "w" )
#        print( str(bpy.context.object.matrix_world.to_quaternion().w), file=f )
#        print( str(bpy.context.object.matrix_world.to_quaternion().x), file=f )
#        print( str(bpy.context.object.matrix_world.to_quaternion().y), file=f )
#        print( str(bpy.context.object.matrix_world.to_quaternion().z), file=f )
#        f.close()

        for col in bpy.data.collections:
            for obj in col.objects:
##                Rotations matrix extraction
                fname = obj.name + ".rotation.csv"
                path = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/"
#                fpath = join( dirname( bpy.data.filepath + "../tf_matrix/" ),fname )
                fpath = join( dirname( path ),fname )
                
                obj.select_set(True)
                q = np.array([], dtype=float)       
                q = np.append(q, obj.matrix_world.to_quaternion().x)
                q = np.append(q, obj.matrix_world.to_quaternion().y)
                q = np.append(q, obj.matrix_world.to_quaternion().z)
                q = np.append(q, obj.matrix_world.to_quaternion().w)
                np.savetxt(fpath, q, delimiter=',')
#            

#               Translation Vector extraction                
                fname = obj.name + ".translation.csv"
                path = "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/tf_matrix/"
#                fpath = join( dirname( bpy.data.filepath + "../tf_matrix/" ),fname )
                fpath = join( dirname( path ),fname )
                
                obj.select_set(True)
                t= np.array([], dtype=float)       
                t = np.append(t, obj.matrix_world.to_translation().x)
                t = np.append(t, obj.matrix_world.to_translation().y)
                t = np.append(t, obj.matrix_world.to_translation().z)
                np.savetxt(fpath, t, delimiter=',')
            
                break
        
#        fname = bpy.context.object.name + ".csv"
#        fpath = join( dirname( bpy.data.filepath ),fname )
#        
#        q = np.array([], dtype=float)
#        q = np.append(q, bpy.context.object.matrix_world.to_quaternion().w)
#        q = np.append(q, bpy.context.object.matrix_world.to_quaternion().x)
#        q = np.append(q, bpy.context.object.matrix_world.to_quaternion().y)
#        q = np.append(q, bpy.context.object.matrix_world.to_quaternion().z)
#        np.savetxt(fpath, q, delimiter=',')
        
import os

import open3d as o3d

from utils import *
from utils import parse_parameters

if __name__ == "__main__":

    opt = parse_parameters()

    folder_path_point_cloud_source = opt["folder_path_point_cloud_source"]
    folder_path_point_cloud_target = opt["folder_path_point_cloud_target"]

    for point_cloud_source_filename, point_cloud_target_filename,  in zip(
                sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target))):
        print(point_cloud_source_filename)
        pcd_source = o3d.io.read_point_cloud(point_cloud_source_filename)
        pcd_target = o3d.io.read_point_cloud(point_cloud_target_filename)
        o3d.io.write_point_cloud(folder_path_point_cloud_source + "/ply/" + point_cloud_source_filename.split(".")[0] + ".ply", pcd_source)
        o3d.io.write_point_cloud(folder_path_point_cloud_target + "/ply/" + point_cloud_target_filename.split(".")[0] + ".ply", pcd_target)

"""
This file will be used to vizualize PCD files in python
"""

import numpy as np
import open3d as o3d


def viz_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    print("asdasd", pcd)
    out_arr = np.asarray(pcd.points)  
    print("output array from input list : ", out_arr)  


if __name__ == '__main__':
    # viz_pcd("")
    # viz_pcd("../pcd_files/2016-06-29-14-37-14_0.bag/1467203834306730.pcd")
    print("I am here")
    pcd = o3d.io.read_point_cloud("1467203834306730.pcd")
    print("Number of points: ", pcd)
    out_arr = np.asarray(pcd.points)  
    print("output array from input list : ", out_arr)  
    # Viz the point clouds
    o3d.visualization.draw_geometries([pcd])

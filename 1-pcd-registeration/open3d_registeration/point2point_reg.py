import argparse
import os

import numpy as np
import open3d as o3d

from utils import (draw_registration_result, execute_global_registration,
                   prepare_dataset)


def point2point_reg(src_dir, trgt_dir, voxel_size):

    trans_init_dir = os.path.join(src_dir, "../o3d_veh_infra_global_reg_transf/")
    
    for src_pcd_filename, trgt_pcd_filename, trans_init_filename in zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir)), 
        sorted(os.listdir(trans_init_dir))):

        print("src pcd filename: ", src_pcd_filename)
        print("trgt pcd filename: ", trgt_pcd_filename)
        print("trans init filename: ", trans_init_filename)
        trans_init = np.load(os.path.join(trans_init_dir,trans_init_filename))
        threshold = 0.02
        # source = o3d.io.read_point_cloud(os.path.join(src_dir,src_pcd_filename))
        # target = o3d.io.read_point_cloud(os.path.join(trgt_dir,trgt_pcd_filename))
        source, target, source_down,\
        target_down, source_fpfh, target_fpfh = prepare_dataset(os.path.join(src_dir,src_pcd_filename), 
                                                                os.path.join(trgt_dir,trgt_pcd_filename),
                                                                voxel_size)
        draw_registration_result(source_down, 
                                target_down, 
                                trans_init)

        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(source_down, target_down, reg_p2p.transformation)
        break

def global_reg(src_dir, trgt_dir, voxel_size):

    for src_pcd_filename, trgt_pcd_filename in zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir))):
        source, target, source_down,\
        target_down, source_fpfh, target_fpfh = prepare_dataset(os.path.join(src_dir,src_pcd_filename), 
                                                                os.path.join(trgt_dir,trgt_pcd_filename),
                                                                voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        print(result_ransac)
        print(type(result_ransac.transformation))
        draw_registration_result(source_down, target_down, result_ransac.transformation)
        np.save(os.path.join(src_dir, "../o3d_veh_infra_global_reg_transf/", os.path.splitext(src_pcd_filename)[0]), 
                result_ransac.transformation)

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do point2point registeration')
    parser.add_argument('--src-dir', type=str, required=True, help='Path to source point clouds directory')
    parser.add_argument('--trgt-dir', type=str, required=True, help='Path to trgt point clouds directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output transformation matrices for each point cloud in kitti format')
    args = parser.parse_args()

    voxel_size = 0.05
    global_reg(args.src_dir, args.trgt_dir, voxel_size)  # means 5cm for this dataset
    point2point_reg(args.src_dir, args.trgt_dir, voxel_size )

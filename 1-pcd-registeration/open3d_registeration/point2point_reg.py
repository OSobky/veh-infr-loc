import argparse
import os

import numpy as np
import open3d as o3d

from utils import draw_registration_result


def point2point_reg(src_dir, trgt_dir):


    for src_pcd_filename, trgt_pcd_filename in zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir))):

        print("src pcd filename: ", src_pcd_filename)
        print("trgt pcd filename: ", trgt_pcd_filename)
        trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1], [0.0, 0.0, 0.0, 1.0]])

        source = o3d.io.read_point_cloud(os.path.join(src_dir,src_pcd_filename))
        target = o3d.io.read_point_cloud(os.path.join(trgt_dir,trgt_pcd_filename))
        draw_registration_result(source, 
                                target, 
                                trans_init)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do point2point registeration')
    parser.add_argument('--src-dir', type=str, required=True, help='Path to source point clouds directory')
    parser.add_argument('--trgt-dir', type=str, required=True, help='Path to trgt point clouds directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output transformation matrices for each point cloud in kitti format')
    args = parser.parse_args()

    point2point_reg(args.src_dir, args.trgt_dir)

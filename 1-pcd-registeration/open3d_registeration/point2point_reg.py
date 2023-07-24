import argparse
import os

import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from utils import (draw_registration_result, execute_global_registration,
                   prepare_dataset, clean_save_pcd, matrix_to_kitti_format, initial_tf_from_gps)


def point2point_reg(src_dir, trgt_dir, gps_dir, imu_dir, voxel_size, vis=False):
    """Point to point registeration using ICP
    Ref: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    
    Compute the P2P registration between two point clouds. then save the transformation matrix in kitti format.

    Arguments:
        src_dir {str} -- Path to source point clouds directory
        trgt_dir {str} -- Path to target point clouds directory
        voxel_size {float} -- voxel size for downsampling
    """

    # Initialize lists to store data
    timestamps = []
    num_correspondences = []
    fitnesses = []
    inlier_rmse = []

    trans_init_dir = os.path.join(src_dir, "../o3d_veh_infra_global_reg_transf/")
    
    for src_pcd_filename, trgt_pcd_filename, gps_filename, imu_filename in zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir)), 
        sorted(os.listdir(gps_dir)), sorted(os.listdir(imu_dir)), 
        # sorted(os.listdir(trans_init_dir))
        ):

        print("src pcd filename: ", src_pcd_filename)
        print("trgt pcd filename: ", trgt_pcd_filename)
        # print("trans init filename: ", trans_init_filename)
        print("gps filename: ", gps_filename)
        print("imu filename: ", imu_filename)
        threshold = 0.02
        # source = o3d.io.read_point_cloud(os.path.join(src_dir,src_pcd_filename))
        # target = o3d.io.read_point_cloud(os.path.join(trgt_dir,trgt_pcd_filename))
        trans_init = initial_tf_from_gps(os.path.join(gps_dir,gps_filename), 
                                         os.path.join(imu_dir,imu_filename))


        source, target, source_down,\
        target_down, source_fpfh, target_fpfh = prepare_dataset(os.path.join(src_dir,src_pcd_filename), 
                                                                os.path.join(trgt_dir,trgt_pcd_filename),
                                                               voxel_size)
        # trans_init = np.load(os.path.join(trans_init_dir,trans_init_filename))
        if vis:
            draw_registration_result(source, 
                                target, 
                                trans_init)

        print("Apply Point2Point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        # Save the transformation matix in KITTI Format
        kitti_tf_matrix = matrix_to_kitti_format(reg_p2p.transformation)
        print(f"TF matix in KITTI Format: {kitti_tf_matrix}")

        # # Save to a text file
        # with open('output.txt', 'w') as f:
        #     f.write(kitti_tf_matrix + '\n')

        # if vis:
        #     draw_registration_result(source, target, reg_p2p.transformation)

        # Store the number of correspondences, fitness, and inlier_rmse
        timestamp = os.path.splitext(src_pcd_filename)[0]
        timestamps.append(timestamp)
        num_correspondences.append(np.asarray(reg_p2p.correspondence_set).shape[0])
        # print(f"Number of correspondences: {np.asarray(reg_p2p.correspondence_set).shape[0]}")
        fitnesses.append(reg_p2p.fitness)
        inlier_rmse.append(reg_p2p.inlier_rmse)
    
    # Save to a DataFrame and then to a CSV file
    data = pd.DataFrame({
        'timestamp': timestamps,
        'num_correspondences': num_correspondences,
        'fitness': fitnesses,
        'inlier_rmse': inlier_rmse
    })
    data.to_csv('icp_data.csv', index=False)

    # Plotting
    fig, axs = plt.subplots(3)
    fig.suptitle('Point2Point ICP Registration Metrics Over Time')

    axs[0].plot(data['timestamp'], data['num_correspondences'], 'tab:orange')
    axs[0].set(xlabel='timestamp', ylabel='Number of Correspondences')

    axs[1].plot(data['timestamp'], data['fitness'], 'tab:green')
    axs[1].set(xlabel='timestamp', ylabel='Fitness')

    axs[2].plot(data['timestamp'], data['inlier_rmse'], 'tab:red')
    axs[2].set(xlabel='timestamp', ylabel='Inlier RMSE')

    for ax in axs.flat:
        ax.label_outer()

    plt.show()
        

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

    # # Clean output directory
    # if os.path.exists(args.output_dir):
    #     for file in os.listdir(args.output_dir):
    #         os.remove(os.path.join(args.output_dir, file))
    # else:       
    #     os.makedirs(args.output_dir)
    
    # # Clean PCD directory
    # # clean_save_pcd(args.src_dir)
    # clean_save_pcd(args.trgt_dir)



    voxel_size = 0.05
    # global_reg(args.src_dir, args.trgt_dir, voxel_size)  # means 5cm for this dataset
    point2point_reg(args.src_dir, args.trgt_dir, 
                    "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/03_gps/04_gps_position_drive/json/matched/" , 
                    "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/04_imu/04_imu_rotations_drive/json/matched/", 
                    voxel_size, vis=False)

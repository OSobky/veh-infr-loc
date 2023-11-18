import argparse
import os

import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

import io
import sys


from utils import (draw_registration_result, execute_global_registration, extract_iterations,
                   prepare_dataset, matrix_to_kitti_format, initial_tf_from_gps, calculate_rmse, 
                   create_experiment_folder, save_draw_registration_result, remove_ground, create_visualizer, stream_registeration_results)


def point2point_online_reg(src_dir, trgt_dir, gps_dir, imu_dir, gt_path, voxel_size, threshold = 3.0 , remove_floor=True, vis=False, stream=False, exp_name="", parent_dir="/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/p2p/online"):
    """Point to point registeration using ICP
    Ref: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    
    Compute the P2P registration between two point clouds. then save the transformation matrix in kitti format.

    Arguments:
        src_dir {str} -- Path to source point clouds directory
        trgt_dir {str} -- Path to target point clouds directory
        voxel_size {float} -- voxel size for downsampling
    """

    # Check if to stream point clouds or not
    if stream:
        # Create a visualizer object
        visualizer = create_visualizer()

    # Initialize lists to store data
    timestamps = []
    num_correspondences = []
    reg_itr = []
    fitnesses = []
    inlier_rmse = []
    translation_rmses = []
    rotation_rmses = []

    # Read ground truth kitti file and save it in an object
    with open(gt_path, 'r') as f:
        gt_list = f.readlines()
    print(len(gt_list))


    trans_init_dir = os.path.join(src_dir, "../o3d_veh_infra_global_reg_transf/")

    # Select frames to save the visualizations
    selected_frames = [50, 150, 250, 350, 450]

    # Create a folder to save the results
    exp_path = create_experiment_folder(parent_dir ,exp_name)
    
    for i, (src_pcd_filename, trgt_pcd_filename, gps_filename, imu_filename) in enumerate(zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir)), 
        sorted(os.listdir(gps_dir)), sorted(os.listdir(imu_dir)), 
        # sorted(os.listdir(trans_init_dir))
        )):

        print("src pcd filename: ", src_pcd_filename)
        print("trgt pcd filename: ", trgt_pcd_filename)
        # print("trans init filename: ", trans_init_filename)
        print("gps filename: ", gps_filename)
        print("imu filename: ", imu_filename)
        
    
        trans_init = initial_tf_from_gps(os.path.join(gps_dir,gps_filename), 
                                         os.path.join(imu_dir,imu_filename))
        

        # print(f"initial Transformation matrix from GPS and IMU: {trans_init}")

        source, target, source_down,\
        target_down, source_fpfh, target_fpfh = prepare_dataset(os.path.join(src_dir,src_pcd_filename), 
                                                                os.path.join(trgt_dir,trgt_pcd_filename),
                                                               voxel_size)
       
        if remove_floor:
            # Remove ground points
            source = remove_ground(source)
            target = remove_ground(target, ground_height=-7.25)



        # turn on the debug mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        # Create a text buffer to capture the output
        log_capture_string = io.StringIO()
        sys.stdout = log_capture_string


        print("Apply Point2Point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Reset the standard output
        sys.stdout = sys.__stdout__

        # Print out the registration result
        # print(reg_p2p)

        # Retrieve the captured logs
        reg_log = log_capture_string.getvalue()
        log_capture_string.close()


        # Usage
        num_iterations = extract_iterations(reg_log)
        print(f"The number of iterations is {num_iterations}") # Output: The number of iterations is 30


        # turn off the debug mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

        
        if vis or i in selected_frames:
            draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)
            # save_draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)
            # load_draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)

        
        ground_truth = np.identity(4)

        if i > 0 and i < 493 :
            # stream point clouds
            if stream:
                stream_registeration_results(source, target, reg_p2p.transformation, exp_path, i, visualizer)
            # Assuming you want to read the i pose, change index 
            ground_truth = np.array(gt_list[i-1].split(), dtype=float)
            ground_truth = np.reshape(ground_truth, (3, 4))

            # Convert to 4x4 transformation matrix
            ground_truth = np.vstack((ground_truth, np.array([0, 0, 0, 1])))

            # Save the transformation matix in KITTI Format
            kitti_tf_matrix = matrix_to_kitti_format(reg_p2p.transformation)
            # print(f"TF matix in KITTI Format: {kitti_tf_matrix}")

            # Save to a text file
            with open(os.path.join(exp_path, 'estimated_tf_kitti.txt') , 'a') as f:
                f.write(kitti_tf_matrix + '\n')

        # print(f"Ground truth transformation matrix: {ground_truth}")
        

        # Calculate RMSEs
        translation_rmse = calculate_rmse(ground_truth[:3, 3], reg_p2p.transformation[:3, 3])
        rotation_rmse = calculate_rmse(ground_truth[:3, :3], reg_p2p.transformation[:3, :3])


        # Store the number of correspondences, fitness, inlier_rmse, translation_rmse, and rotation_rmse
        timestamp = os.path.splitext(src_pcd_filename)[0]
        timestamps.append(timestamp)
        num_correspondences.append(np.asarray(reg_p2p.correspondence_set).shape[0])
        reg_itr.append(num_iterations)
        fitnesses.append(reg_p2p.fitness)
        inlier_rmse.append(reg_p2p.inlier_rmse)
        translation_rmses.append(translation_rmse)
        rotation_rmses.append(rotation_rmse)


    # distroy the visualizer object
    if stream:
        visualizer.destroy_window()
    
    # Save to a DataFrame and then to a CSV file
    data = pd.DataFrame({
        'timestamp': timestamps,
        'num_correspondences': num_correspondences,
        'reg_itr': reg_itr,
        'fitness': fitnesses,
        'inlier_rmse': inlier_rmse,
        'translation_rmse': translation_rmses,
        'rotation_rmse': rotation_rmses

    })
    data.to_csv(os.path.join(exp_path,'icp_data.csv'), index=False)

    # Plotting
    fig, axs = plt.subplots(6)
    fig.suptitle('Point2Point ICP Registration Metrics Over Time with a Threshold of {}'.format(threshold))

    axs[0].plot(data['timestamp'], data['reg_itr'], 'tab:blue')
    axs[0].set(xlabel='timestamp', ylabel='#_Itr')

    axs[1].plot(data['timestamp'], data['num_correspondences'], 'tab:orange')
    axs[1].set(xlabel='timestamp', ylabel='#_Crsp')

    axs[2].plot(data['timestamp'], data['fitness'], 'tab:green')
    axs[2].set(xlabel='timestamp', ylabel='Fitness')

    axs[3].plot(data['timestamp'], data['inlier_rmse'], 'tab:red')
    axs[3].set(xlabel='timestamp', ylabel='Inlier_RMSE')

    axs[4].plot(data['timestamp'], data['translation_rmse'], 'tab:blue')
    axs[4].set(xlabel='timestamp', ylabel='Tsl_RMSE')

    axs[5].plot(data['timestamp'], data['rotation_rmse'], 'tab:purple')
    axs[5].set(xlabel='timestamp', ylabel='Rot_RMSE')


    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(exp_path, 'icp_metrics.png'))

    if vis:
        plt.show()
        
        

def point2point_offline_reg(src_dir, local_map_path, gps_dir, imu_dir, gt_path, voxel_size, threshold = 3.0 , remove_floor=True, vis=False, exp_name="", parent_dir="/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/p2p/offline"):
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
    reg_itr = []
    fitnesses = []
    inlier_rmse = []
    translation_rmses = []
    rotation_rmses = []

    # Read ground truth kitti file and save it in an object
    with open(gt_path, 'r') as f:
        gt_list = f.readlines()
    print(len(gt_list))


    trans_init_dir = os.path.join(src_dir, "../o3d_veh_infra_global_reg_transf/")

    # Select frames to save the visualizations
    selected_frames = [50, 150, 250, 350, 450]

    # Create a folder to save the results
    exp_path = create_experiment_folder(parent_dir, exp_name)
    
    for i, (src_pcd_filename, gps_filename, imu_filename) in enumerate(zip(
        sorted(os.listdir(src_dir)), 
        sorted(os.listdir(gps_dir)), sorted(os.listdir(imu_dir)), 
        # sorted(os.listdir(trans_init_dir))
        )):

        print("src pcd filename: ", src_pcd_filename)
        # print("trans init filename: ", trans_init_filename)
        print("gps filename: ", gps_filename)
        print("imu filename: ", imu_filename)
        
    
        trans_init = initial_tf_from_gps(os.path.join(gps_dir,gps_filename), 
                                         os.path.join(imu_dir,imu_filename))
        

        # print(f"initial Transformation matrix from GPS and IMU: {trans_init}")

        source, target, source_down,\
        target_down, source_fpfh, target_fpfh = prepare_dataset(os.path.join(src_dir,src_pcd_filename), 
                                                                local_map_path,
                                                               voxel_size)
       
        if remove_floor:
            # Remove ground points
            source = remove_ground(source)
            target = remove_ground(target, ground_height=-7.25)



        # turn on the debug mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        # Create a text buffer to capture the output
        log_capture_string = io.StringIO()
        sys.stdout = log_capture_string


        print("Apply Point2Point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Reset the standard output
        sys.stdout = sys.__stdout__

        # Print out the registration result
        # print(reg_p2p)

        # Retrieve the captured logs
        reg_log = log_capture_string.getvalue()
        log_capture_string.close()


        # Usage
        num_iterations = extract_iterations(reg_log)
        print(f"The number of iterations is {num_iterations}") # Output: The number of iterations is 30


        # turn off the debug mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

        
        if vis or i in selected_frames:
            draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)
            # save_draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)
            # load_draw_registration_result(source, target, reg_p2p.transformation, exp_path, i)

        
        ground_truth = np.identity(4)

        if i > 0 and i < 493 :
            # Assuming you want to read the i pose, change index 
            ground_truth = np.array(gt_list[i-1].split(), dtype=float)
            ground_truth = np.reshape(ground_truth, (3, 4))

            # Convert to 4x4 transformation matrix
            ground_truth = np.vstack((ground_truth, np.array([0, 0, 0, 1])))

            # Save the transformation matix in KITTI Format
            kitti_tf_matrix = matrix_to_kitti_format(reg_p2p.transformation)
            # print(f"TF matix in KITTI Format: {kitti_tf_matrix}")

            # Save to a text file
            with open(os.path.join(exp_path, 'estimated_tf_kitti.txt') , 'a') as f:
                f.write(kitti_tf_matrix + '\n')

        # print(f"Ground truth transformation matrix: {ground_truth}")
        

        # Calculate RMSEs
        translation_rmse = calculate_rmse(ground_truth[:3, 3], reg_p2p.transformation[:3, 3])
        rotation_rmse = calculate_rmse(ground_truth[:3, :3], reg_p2p.transformation[:3, :3])


        # Store the number of correspondences, fitness, inlier_rmse, translation_rmse, and rotation_rmse
        timestamp = os.path.splitext(src_pcd_filename)[0]
        timestamps.append(timestamp)
        num_correspondences.append(np.asarray(reg_p2p.correspondence_set).shape[0])
        reg_itr.append(num_iterations)
        fitnesses.append(reg_p2p.fitness)
        inlier_rmse.append(reg_p2p.inlier_rmse)
        translation_rmses.append(translation_rmse)
        rotation_rmses.append(rotation_rmse)



        
    
    # Save to a DataFrame and then to a CSV file
    data = pd.DataFrame({
        'timestamp': timestamps,
        'num_correspondences': num_correspondences,
        'reg_itr': reg_itr,
        'fitness': fitnesses,
        'inlier_rmse': inlier_rmse,
        'translation_rmse': translation_rmses,
        'rotation_rmse': rotation_rmses

    })
    data.to_csv(os.path.join(exp_path,'icp_data.csv'), index=False)

    # Plotting
    fig, axs = plt.subplots(6)
    fig.suptitle('Point2Point ICP Registration Metrics Over Time with a Threshold of {}'.format(threshold))

    axs[0].plot(data['timestamp'], data['reg_itr'], 'tab:blue')
    axs[0].set(xlabel='timestamp', ylabel='#_Itr')

    axs[1].plot(data['timestamp'], data['num_correspondences'], 'tab:orange')
    axs[1].set(xlabel='timestamp', ylabel='#_Crsp')

    axs[2].plot(data['timestamp'], data['fitness'], 'tab:green')
    axs[2].set(xlabel='timestamp', ylabel='Fitness')

    axs[3].plot(data['timestamp'], data['inlier_rmse'], 'tab:red')
    axs[3].set(xlabel='timestamp', ylabel='Inlier_RMSE')

    axs[4].plot(data['timestamp'], data['translation_rmse'], 'tab:blue')
    axs[4].set(xlabel='timestamp', ylabel='Tsl_RMSE')

    axs[5].plot(data['timestamp'], data['rotation_rmse'], 'tab:purple')
    axs[5].set(xlabel='timestamp', ylabel='Rot_RMSE')


    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(os.path.join(exp_path, 'icp_metrics.png'))

    if vis:
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do point2point registeration')
    parser.add_argument('--src-dir', type=str, required=True, help='Path to source point clouds directory')
    parser.add_argument('--trgt', type=str, required=True, help='Path to trgt point clouds directory or Path of the target local map')
    parser.add_argument("--offline", action="store_true", help="Do offline registeration (aginest a local map)")
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

    #  Loop over thershoulds
    # for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:
    for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:
        # Run ICP
        voxel_size = 0.05
        # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        if args.offline:
            point2point_offline_reg(args.src_dir, args.trgt, 
                        "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/03_gps/04_gps_position_drive/json/matched/" , 
                        "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/04_imu/04_imu_rotations_drive/json/matched/",
                        "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt",
                        voxel_size, 
                        threshold,
                        remove_floor=False,
                        vis=False,
                        exp_name=f"floor_not_removed_{threshold}_threshold")
        else:
            point2point_online_reg(args.src_dir, args.trgt, 
                            "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/03_gps/04_gps_position_drive/json/matched/" , 
                            "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/04_imu/04_imu_rotations_drive/json/matched/",
                            "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt",
                            voxel_size, 
                            threshold,
                            remove_floor=False,
                            vis=False,
                            stream=False,
                            exp_name=f"floor_not_removed_{threshold}_threshold")

    # evo("/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/veh-infra-gt.txt",
    #     "estimated_tf_kitti.txt")

    
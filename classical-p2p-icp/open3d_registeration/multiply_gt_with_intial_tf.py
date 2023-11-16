import numpy as np
from utils import create_experiment_folder, draw_registration_result, initial_tf_from_gps, matrix_to_kitti_format
import os
import open3d as o3d


def apply_transformation_to_kitti_file(file_path, gps_dir, imu_dir):
    """
    Function to read a file in KITTI format and multiply each transformation matrix
    with an initial transformation matrix.

    Args:
        file_path (str): Path to the file in KITTI format
        gps_folder (str): Path to the GPS data folder
        imu_folder (str): Path to the IMU data folder
    """

    # Read ground truth kitti file and save it in an object
    with open(file_path, 'r') as f:
        gt_list = f.readlines()

    # Initialize a list to store the transformed matrices
    fixed_gt = []

    for i, (gps_filename, imu_filename) in enumerate(zip(
                                    sorted(os.listdir(gps_dir)), sorted(os.listdir(imu_dir)))):
        if i >= 493:
            break

        print("gps filename: ", gps_filename)
        print("imu filename: ", imu_filename)

        if i > 0:
            trans_init = initial_tf_from_gps(os.path.join(gps_dir,gps_filename), 
                                         os.path.join(imu_dir,imu_filename))
        
            # Assuming you want to read the i pose, change index 
            ground_truth = np.array(gt_list[i-1].split(), dtype=float)
            ground_truth = np.reshape(ground_truth, (3, 4))

            # Convert to 4x4 transformation matrix
            ground_truth = np.vstack((ground_truth, np.array([0, 0, 0, 1])))

            # Multiply the transformation matrix by the initial transformation matrix
            # transformed_matrix = np.dot(trans_init, ground_truth)
            transformed_matrix = ground_truth @ trans_init 

            # Append the transformed matrix to the list
            fixed_gt.append(transformed_matrix)

    # Save the transformed matrices to a text file
    with open(os.path.join(os.path.dirname(file_path), f"{os.path.splitext(os.path.basename(file_path))[0]}_fixed_gt.txt") , 'w') as f:
        for matrix in fixed_gt:
            kitti_tf_matrix = matrix_to_kitti_format(matrix)
            f.write(kitti_tf_matrix + '\n')

def transform_and_viz(src_dir, trgt_dir, kitti_path):
    # Read ground truth kitti file and save it in an object
    with open(kitti_path, 'r') as f:
        tf_list = f.readlines()

    # Create a folder to save the results
    exp_path = create_experiment_folder("/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/")

    for i, (src_pcd_filename, trgt_pcd_filename) in enumerate(zip(
        sorted(os.listdir(src_dir)), sorted(os.listdir(trgt_dir)),
        )):
        if i >= 493:
            break

        print("src pcd filename: ", src_pcd_filename)
        print("trgt pcd filename: ", trgt_pcd_filename)

        # Assuming you want to read the i pose, change index 
        tf_matrix = np.array(tf_list[i-1].split(), dtype=float)
        tf_matrix = np.reshape(tf_matrix, (3, 4))

        # Convert to 4x4 transformation matrix
        tf_matrix = np.vstack((tf_matrix, np.array([0, 0, 0, 1])))

        # Read source and target point clouds
        source = o3d.io.read_point_cloud(os.path.join(src_dir,src_pcd_filename), remove_nan_points=True, remove_infinite_points=True)
        target = o3d.io.read_point_cloud(os.path.join(trgt_dir,trgt_pcd_filename), remove_nan_points=True, remove_infinite_points=True)

        draw_registration_result(source, target, tf_matrix, exp_path, i)



        

if __name__ == "__main__":
    # file_path = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/veh-infra-gt.txt"
    file_path = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/experiments/abilation/2023-10-09_00-00-16/transformed_cleaned_dir_poses_kitti.txt"
    gps_dir = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/03_gps/04_gps_position_drive/json/matched/"
    imu_dir = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/04_imu/04_imu_rotations_drive/json/matched/"

    apply_transformation_to_kitti_file(file_path, gps_dir, imu_dir)

    # src_dir = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched"
    # trgt_dir = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/matched"
    # kitti_path = "/mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/tf_matrix/kitti/fixed_gt.txt"

    # transform_and_viz(src_dir, trgt_dir, kitti_path)
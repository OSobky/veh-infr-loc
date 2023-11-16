import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd
import utm
from scipy.spatial.transform import Rotation as R


def get_initial_transformation_matrix(lat, long, alt, quaternion_x, quaternion_y, quaternion_z, quaternion_w):
    float_formatter = "{:.8f}".format
    np.set_printoptions(suppress=True, formatter={'float_kind': float_formatter})

    # 1. transformation matrix (vehicle lidar to infrastructure lidar)
    # 1.1 vehicle lidar pose
    utm_east_vehicle_lidar, utm_north_vehicle_lidar, zone, _ = utm.from_latlon(lat, long)
    utm_east_vehicle_lidar = utm_east_vehicle_lidar
    utm_north_vehicle_lidar = utm_north_vehicle_lidar
    altitude_vehicle_lidar = alt

    rotation_x_vehicle_lidar = quaternion_x
    rotation_y_vehicle_lidar = quaternion_y
    rotation_z_vehicle_lidar = quaternion_z
    rotation_w_vehicle_lidar = quaternion_w
    rotation_roll_vehicle_lidar, rotation_pitch_vehicle_lidar, rotation_yaw_vehicle_lidar = R.from_quat(
        [rotation_x_vehicle_lidar, rotation_y_vehicle_lidar, rotation_z_vehicle_lidar,
         rotation_w_vehicle_lidar]).as_euler('xyz', degrees=True)
    rotation_yaw_vehicle_lidar = 0

    print("euler angles (lidar vehicle): ",
          [rotation_roll_vehicle_lidar, rotation_pitch_vehicle_lidar, rotation_yaw_vehicle_lidar])
    rotation_yaw_vehicle_lidar = -rotation_yaw_vehicle_lidar

    # 1.2 infrastructure lidar pose
    utm_east_s110_lidar_ouster_south = 695308.460000000 - 0.5
    utm_north_s110_lidar_ouster_south = 5347360.569000000 + 2.5
    # altitude_s110_lidar_ouster_south = 534.3500000000000 + 1.0

    # Omar's Coooodeee
    # =====================================================================
    # What we will be done here?
    #   - Trying to fit the whole scene of the traffic light manually 
    #   - After viz with RViZ i can see that it needs to be lowered by 0.5 meters

    altitude_s110_lidar_ouster_south = 534.3500000000000 + 1.0 + 0.5

    # =====================================================================

    rotation_roll_s110_lidar_ouster_south = 0  # 1.79097398157454 # pitch
    rotation_pitch_s110_lidar_ouster_south = 1.1729642881072  # roll
    rotation_yaw_s110_lidar_ouster_south = 172  # 172.693672075377

    translation_vehicle_lidar_to_s110_lidar_ouster_south = np.array(
        [utm_east_s110_lidar_ouster_south - utm_east_vehicle_lidar,
         utm_north_s110_lidar_ouster_south - utm_north_vehicle_lidar,
         altitude_s110_lidar_ouster_south - altitude_vehicle_lidar], dtype=float)
    print("translation vehicle lidar to s110_lidar_ouster_south: ",
          translation_vehicle_lidar_to_s110_lidar_ouster_south)

    rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south = R.from_rotvec(
        [rotation_roll_s110_lidar_ouster_south - rotation_roll_vehicle_lidar,
         rotation_pitch_s110_lidar_ouster_south - rotation_pitch_vehicle_lidar,
         rotation_yaw_s110_lidar_ouster_south - rotation_yaw_vehicle_lidar],
        degrees=True).as_matrix().T

    print("rotation yaw final: ", str(rotation_yaw_s110_lidar_ouster_south - rotation_yaw_vehicle_lidar))

    translation_vector_rotated = np.matmul(rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south,
                                           translation_vehicle_lidar_to_s110_lidar_ouster_south)
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south
    transformation_matrix[0:3, 3] = -translation_vector_rotated
    transformation_matrix[3, 3] = 1.0
    print("transformation matrix: vehicle_lidar to s110_lidar_ouster_south")
    print(repr(transformation_matrix))

    # Omar's Coooodeee
    # =====================================================================
    # What we will be done here?
    #   - Trying to fit the whole scene of the traffic light manually 
    #   - After viz with RViZ i can see that it needs to be rotated around the X-axis by 1 or 2 degrees

    rotate_around_x = R.from_euler('x', -1.5, degrees=True)
    transformation_matrix_rotate_x= np.zeros((4, 4))
    transformation_matrix_rotate_x[0:3, 0:3] = rotate_around_x.as_matrix()
    transformation_matrix_rotate_x[3, 3] = 1.0

    transformation_matrix = np.matmul(transformation_matrix, transformation_matrix_rotate_x.T)


    # Omar's Coooodeee
    # =====================================================================
    # What we will be done here?
    #   - after manually register two PCD with Belender I got the transformation matrix I will apply it for all frames

    # rotate_blender = R.from_euler('xyz', [0.6,0.4,0.9], degrees=True)
    # transformation_matrix_blender= np.zeros((4, 4))
    
    # transformation_matrix_blender[0:3, 3]= -np.array([-0.36, 1.03, -0.1])
    # print("transformation matrix:after adding translation")
    # print(repr(transformation_matrix_blender))
    # transformation_matrix_blender[0:3, 0:3] = rotate_blender.as_matrix()
    # transformation_matrix_blender[3, 3] = 1.0

    # transformation_matrix = np.matmul(transformation_matrix, transformation_matrix_blender)

    print("transformation matrix: vehicle_lidar to s110_lidar_ouster_south with blender matrix")
    print(repr(transformation_matrix))

    # =====================================================================
    # transformation_matrix = np.round(transformation_matrix, 8)
    # transformation_matrix = np.linalg.inv(transformation_matrix)
    # print(repr(transformation_matrix))
    return transformation_matrix


def read_point_cloud(input_file_path):
    lines = None
    with open(input_file_path, "r") as reader:
        lines = reader.readlines()
    header = lines[:11]
    point_cloud_array = np.array(pd.read_csv(input_file_path, sep=' ', skiprows=11, dtype=float).values)[:, :4]
    return point_cloud_array, header


def write_point_cloud_with_intensities(output_file_path, point_cloud_array, header):
    # update num points
    header[6] = "WIDTH " + str(len(point_cloud_array)) + "\n"
    header[7] = "HEIGHT 1" + "\n"
    header[9] = "POINTS " + str(len(point_cloud_array)) + "\n"
    with open(output_file_path, 'w') as writer:
        for header_line in header:
            writer.write(header_line)
    df = pd.DataFrame(point_cloud_array)
    df.to_csv(output_file_path, sep=" ", header=False, mode='a', index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument('--input_folder_path_gps', default='input/images',
                           help='Input folder path to gps data')
    args = argparser.parse_args()

    for file_name in sorted(os.listdir(args.input_folder_path_gps)):
        transformation_matrix = get_initial_transformation_matrix(os.path.join(args.input_folder_path_gps, file_name))

        # load point cloud vehicle
        point_cloud_vehicle_array_with_intensities, point_cloud_vehicle_header = read_point_cloud(
            "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/05_test_frames/1667908155_353740000_vehicle_lidar_robosense_ascii.pcd")
        vehicle_intensities = point_cloud_vehicle_array_with_intensities[:, 3]
        # normalize intensities
        vehicle_intensities *= (1 / vehicle_intensities.max())
        # load point cloud infrastructure
        point_cloud_infrastructure_array_with_intensities, point_cloud_infrastructure_header = read_point_cloud(
            "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/05_test_frames/1667908155_367814744_s110_lidar_ouster_south_ascii.pcd")

        point_cloud_infrastructure_array_with_intensities = point_cloud_infrastructure_array_with_intensities[
            ~np.all(point_cloud_infrastructure_array_with_intensities[:, :3] == 0.0, axis=1)]

        infrastructure_intensities = point_cloud_infrastructure_array_with_intensities[:, 3]
        infrastructure_intensities *= (1 / infrastructure_intensities.max())

        write_point_cloud_with_intensities(
            "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/05_test_frames/1667908155_367814744_s110_lidar_ouster_south_filtered_ascii.pcd",
            point_cloud_infrastructure_array_with_intensities, point_cloud_vehicle_header)

        # set infrastructure lidar color to light red
        # pcd_s110_lidar_ouster_south.paint_uniform_color((255 / 255.0, 64 / 255.0, 64 / 255.0))

        # transform point cloud vehicle to infrastructure
        # pcd_vehicle_transformed = o3d.geometry.PointCloud()
        # add ones as column vector
        # pcd_vehicle_array = np.array(pcd_vehicle.points)  # n x 3
        # remove nans
        nan_indices = ~np.isnan(point_cloud_vehicle_array_with_intensities).any(axis=1)
        point_cloud_vehicle_array = point_cloud_vehicle_array_with_intensities[nan_indices, :3]
        vehicle_intensities = vehicle_intensities[nan_indices]
        ones_col = np.ones((point_cloud_vehicle_array.shape[0], 1))
        pcd_vehicle_array_homogeneous = np.hstack([point_cloud_vehicle_array, ones_col])
        # pcd_vehicle_array_homogeneous = pcd_vehicle_array_homogeneous.T
        # print("shape: ",pcd_vehicle_array_homogeneous.shape)
        point_cloud_vehicle_array_transformed = np.matmul(transformation_matrix, pcd_vehicle_array_homogeneous.T).T[:,
                                                :3]
        # pcd_vehicle_transformed.points = o3d.utility.Vector3dVector(pcd_vehicle_array_transformed)
        # pcd_vehicle_transformed.point["positions"] = o3d.core.Tensor(pcd_vehicle_array_transformed, dtype, device)
        # pcd_vehicle_transformed.point["intensities"] = o3d.core.Tensor(vehicle_intensities, dtype, device)
        # set vehicle lidar colors to green
        # pcd_vehicle_transformed.paint_uniform_color((64 / 255.0, 255 / 255.0, 64 / 255.0))
        # save transformed vehicle point cloud
        # o3d.t.io.write_point_cloud("/home/providentia/Downloads/test_frames/point_cloud_vehicle_transformed.pcd",
        #                            pcd_vehicle_transformed, write_ascii=True)
        # add intensities to transformed points
        # make nx1 dimension from (n,)
        vehicle_intensities = vehicle_intensities.reshape((vehicle_intensities.shape[0], 1))
        point_cloud_vehicle_transformed_with_intensities = np.hstack(
            [point_cloud_vehicle_array_transformed, vehicle_intensities])
        write_point_cloud_with_intensities(
            "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/05_test_frames/point_cloud_vehicle_transformed.pcd",
            point_cloud_vehicle_transformed_with_intensities, point_cloud_vehicle_header)

        # merge point cloud
        # pcd_merged = o3d.geometry.PointCloud()

        # stack points
        points_stacked = np.vstack(
            [point_cloud_vehicle_transformed_with_intensities, point_cloud_infrastructure_array_with_intensities])
        # pcd_merged.points = o3d.utility.Vector3dVector(points_stacked)
        # stack colors
        # colors_stacked = np.vstack([pcd_vehicle_transformed.colors, pcd_s110_lidar_ouster_south.colors])
        # pcd_merged.colors = o3d.utility.Vector3dVector(colors_stacked)
        # save point cloud
        write_point_cloud_with_intensities(
            "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/05_test_frames/merged_point_cloud.pcd",
            points_stacked,
            point_cloud_vehicle_header)

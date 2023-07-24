import json
import os
from argparse import ArgumentParser
import numpy as np

from utils import read_point_cloud, filter_point_cloud, \
    write_point_cloud_with_intensities

from point_cloud_registration_pcd import register_point_clouds

from package.point_cloud_registration.scripts.transform_and_merge import get_initial_transformation_matrix

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--file_path_point_cloud_source',
                        default='/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_valeo_north_west/source.pcd',
                        help='file path of source point cloud (will be transformed to target point cloud frame)')
    parser.add_argument('--file_path_point_cloud_target',
                        default='/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_south/target.pcd',
                        help='file path of target point cloud (remains static and will not be transformed)')
    parser.add_argument('--initial_voxel_size', default=2, help='initial voxel size')
    parser.add_argument('--continuous_voxel_size', default=2, help='continuous voxel size')
    parser.add_argument('--output_file_path_point_cloud', default="output",
                        help='Output file path to save point cloud (default: output/output.pcd)')

    opt = vars(parser.parse_args())
    file_path_point_cloud_source = opt["file_path_point_cloud_source"]
    file_path_point_cloud_target = opt["file_path_point_cloud_target"]
    initial_voxel_size = float(opt["initial_voxel_size"])
    continuous_voxel_size = float(opt["continuous_voxel_size"])
    output_file_path_point_cloud = opt["output_file_path_point_cloud"]

    point_cloud_source, header_source = read_point_cloud(file_path_point_cloud_source)
    point_cloud_target, header_target = read_point_cloud(file_path_point_cloud_target)

    point_cloud_source = filter_point_cloud(point_cloud_source)
    point_cloud_target = filter_point_cloud(point_cloud_target)


    # transformation_matrix = np.array([[0.891, 0.001, 0.002, 400.1],
    #                                   [0.5, 0.892, 0.0, 200.2],
    #                                   [0.3, 0.3, 0.893, 1.3],
    #                                   [0.0, 0.0, 0.0, 1.0]], dtype=float)

    # store intensities
    intensities_source = point_cloud_source[:, 3]
    one_column = np.ones((len(point_cloud_source), 1), dtype=float)
    point_cloud_source_homogeneous = np.concatenate((point_cloud_source[:, 0:3], one_column), axis=1)

    # point_cloud_target = np.matmul(transformation_matrix, point_cloud_source_homogeneous.T).T
    # restore intensities
    # point_cloud_target[:, 3] = intensities

    # add color to source
    colors = np.ones((len(point_cloud_source), 1), dtype=float)
    colors[:, 0] = 64 << 16 | 255 << 8 | 64
    point_cloud_source = np.concatenate((point_cloud_source, colors), axis=1)
    # write_point_cloud_with_intensities(
    #     "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/test/pcd_source_with_colors.pcd",
    #     point_cloud_source, header_source)

    # add color to target
    colors = np.ones((len(point_cloud_target), 1), dtype=float)
    colors[:, 0] = 255 << 16 | 64 << 8 | 64
    point_cloud_target = np.concatenate((point_cloud_target, colors), axis=1)
    # write_point_cloud_with_intensities(
    #     "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/test/pcd_target_with_colors.pcd",
    #     point_cloud_target, header_source)

    # stack/merge points
    points_stacked = np.vstack(
        [point_cloud_source, point_cloud_target])
    write_point_cloud_with_intensities(
        "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/test/pcd_before_registration.pcd",
        points_stacked, header_source)

    gps_json = json.load(open("/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/03_gps/04_gps_position_drive/json/matched/1667908120_000000000.json"))
    imu_json = json.load(open("/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/04_imu/04_imu_rotations_drive/json/matched/1667908120_000000000.json"))

    transformation_matrix_initial = get_initial_transformation_matrix(float(gps_json["lat"]),
                                                                      float(gps_json["long"]),
                                                                      float(gps_json["alt"]),
                                                                      float(imu_json["quaternion_x"]),
                                                                      float(imu_json["quaternion_y"]),
                                                                      float(imu_json["quaternion_z"]),
                                                                      float(imu_json["quaternion_w"]))

    # apply initial transformation
    point_cloud_source_before_refinement = np.matmul(transformation_matrix_initial, point_cloud_source_homogeneous.T).T
    point_cloud_source_before_refinement[:, 3] = intensities_source

    # save point cloud after initial transformation (before refinement)
    # add color to point cloud
    colors = np.ones((len(point_cloud_source_before_refinement), 1), dtype=float)
    colors[:, 0] = 64 << 16 | 255 << 8 | 64
    point_cloud_source_before_refinement = np.concatenate((point_cloud_source_before_refinement, colors), axis=1)

    points_stacked = np.vstack(
        [point_cloud_target, point_cloud_source_before_refinement])
    write_point_cloud_with_intensities(
        "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/test/pcd_before_refinement.pcd",
        points_stacked, header_source)


    do_initial_registration = False
    transformation_matrix, inlier_rmse_best, fitness_best = register_point_clouds(do_initial_registration,
                                                                                  point_cloud_source,
                                                                                  point_cloud_target, 0,
                                                                                  transformation_matrix_initial,
                                                                                  initial_voxel_size,
                                                                                  continuous_voxel_size)
    intensities_target = point_cloud_target[:, 3]
    one_column = np.ones((len(point_cloud_target), 1), dtype=float)
    point_cloud_target_homogeneous = np.concatenate((point_cloud_target[:, 0:3], one_column), axis=1)

    point_cloud_registered = np.matmul(np.linalg.inv(transformation_matrix), point_cloud_target_homogeneous.T).T
    # restore intensities
    point_cloud_registered[:, 3] = intensities_target

    # add color to registered
    colors = np.ones((len(point_cloud_registered), 1), dtype=float)
    colors[:, 0] = 255 << 16 | 64 << 8 | 64
    point_cloud_registered = np.concatenate((point_cloud_registered, colors), axis=1)

    points_stacked = np.vstack(
        [point_cloud_source, point_cloud_registered])
    write_point_cloud_with_intensities(
        "/mnt/hdd_data1/34_vehicle_infrastructure_recordings/01_scene_01/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/test/pcd_after_refinement.pcd",
        points_stacked, header_source)

import json
import os
import sys
import time

import numpy as np

from transform_and_merge import get_initial_transformation_matrix
from utils import *
from utils import (create_intensity_point_cloud2_msg, filter_point_cloud,
                   parse_parameters, read_point_cloud,
                   write_point_cloud_with_intensities)

np.set_printoptions(suppress=True)
import threading
from argparse import ArgumentParser

import message_filters
import open3d as o3d
import pandas as pd
import rospy
import sensor_msgs.point_cloud2 as pc2
import utm
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# Register each point cloud and find the optimal transformation among all input .pcd files -> optimal result
# NOTE: use the final transformation matrix in merge_point_clouds.py

# Example: python point_cloud_registration_pcd.py --folder_path_point_cloud_source /home/providentia/Downloads/test_frames/lidar_vehicle_point_cloud/ --folder_path_point_cloud_target /home/providentia/Downloads/test_frames/lidar_infrastructure_point_cloud/ --init_voxel_size 2 --continuous_voxel_size 2 --output_folder_path_registered_point_clouds /home/providentia/Downloads/test_frames/registered_point_clouds/ --save_registered_point_clouds --publish_registered_point_clouds
# NOTE: set the initial transformation matrix in line 195!

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_point_cloud(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_initial_registration(source_down, target_down, source_fpfh,
                                 target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, voxel_size, trans_init):
    use_point_to_plane = False
    distance_threshold = voxel_size * 0.4
    if use_point_to_plane:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    return result


def get_xyzi_points(cloud_array, remove_nans=True, dtype=float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    points[..., 3] = cloud_array['intensity']

    return points


def register_point_clouds(do_initial_registration, point_cloud_source, point_cloud_target, idx_file,
                          transformation_matrix, initial_voxel_size, continuous_voxel_size):
    num_initial_registration_loops = 4

    inlier_rmse_best = sys.maxsize
    fitness_best = 0

    point_cloud_array_source = np.array(point_cloud_source)
    point_cloud_array_target = np.array(point_cloud_target)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_cloud_array_source[:, 0:3])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(point_cloud_array_target[:, 0:3])

    voxel_size = continuous_voxel_size
    if do_initial_registration:
        for i in range(num_initial_registration_loops):
            # NOTE: the smaller the voxel_size the more accurate is the initial transformation matrix
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_point_cloud(source, target,
                                                                                                     initial_voxel_size)
            initial_registration_result = execute_initial_registration(source_down, target_down, source_fpfh,
                                                                       target_fpfh, initial_voxel_size)
            if initial_registration_result.inlier_rmse < inlier_rmse_best and initial_registration_result.fitness > fitness_best:
                transformation_matrix = initial_registration_result.transformation
                inlier_rmse_best = initial_registration_result.inlier_rmse
                fitness_best = initial_registration_result.fitness
                print("==========================")
                print("Better transformation matrix found (using initial registration):\n ",
                      repr(transformation_matrix))
                print("With frame index: %d" % idx_file)
                print("With better RMSE: %.4f" % inlier_rmse_best)
                print("With better fitness: %.4f" % fitness_best)
                print("==========================")

            if transformation_matrix is not None:
                continuous_registration_result = refine_registration(source_down, target_down, voxel_size,
                                                                     transformation_matrix)
                if continuous_registration_result.inlier_rmse < inlier_rmse_best and continuous_registration_result.fitness > fitness_best:
                    inlier_rmse_best = continuous_registration_result.inlier_rmse
                    fitness_best = continuous_registration_result.fitness
                    transformation_matrix = continuous_registration_result.transformation
                    print("==========================")
                    print("Better transformation matrix found (using continuous registration):\n ",
                          repr(transformation_matrix))
                    print("With frame index: %d" % idx_file)
                    print("With better RMSE: %.4f" % inlier_rmse_best)
                    print("With better fitness: %.4f" % fitness_best)
                    print("==========================")

    else:
        # TODO temporary hard code initial transformation matrix
        # initial transformation matrix ouster_north to ouster_south
        # transformation_matrix = np.array([[9.68911602e-01, -2.47355442e-01, -5.05895822e-03, 2.07276299e+00],
        #                                    [2.47342195e-01, 9.68923216e-01, -3.10486184e-03, -1.35403183e+01],
        #                                    [5.66974654e-03, 1.75704282e-03, 9.99982383e-01, 1.35447590e-01],
        #                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
        #                                   dtype=float)
        # initial transformation matrix ouster_south to ouster_north
        # transformation_matrix = np.linalg.inv(transformation_matrix)

        # initial transformation matrix valeo_north_west to ouster_south
        # transformation_matrix = np.array([[0.75184951, 0.64600574, 0.13190484, -0.1206516],
        #                                   [-0.63314934, 0.76321442, -0.12894047, 1.24191049],
        #                                   [-0.18396796, 0.01342837, 0.98284051, -0.05209358],
        #                                   [0.0, 0.0, 0.0, 1.0]], dtype=float)

        # initial transformation matrix: vehicle_lidar_robosense to s110_lidar_ouster_south
        # with -1
        # transformation_matrix = np.array([[-0.99005784, 0.13949302, 0.01808788, 21.26957863],
        #                                   [-0.13925646, -0.99016097, 0.01374358, 7.85061464],
        #                                   [0.01982705, 0.01108808, 0.99974194, -5.84412526],
        #                                   [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
        # transformation_matrix = np.array([[-0.99005784, 0.13949302, 0.01808788, 21.26957864],
        #                                   [-0.13925646, -0.99016097, 0.01374358, 7.85061463],
        #                                   [0.01982705, 0.01108808, 0.99974194, -5.84412526],
        #                                   [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
        # no change
        # transformation_matrix = np.array([[-0.99005784, 0.13949302, 0.01808788, 21.40907165],
        #                                   [-0.13925646, -0.99016097, 0.01374358, 6.86045366],
        #                                   [0.01982705, 0.01108808, 0.99974194, -5.83303718],
        #                                   [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
        print("Initial transformation_matrix: \n", repr(transformation_matrix))

        # initial transformation matrix s110_lidar_valeo_north_west to s110_lidar_ouster_north
        # NOTE: used from ouster_south
        # transformation_matrix = np.array([[9.68911602e-01, -2.47355442e-01, -5.05895822e-03, 2.07276299e+00],
        #                                    [2.47342195e-01, 9.68923216e-01, -3.10486184e-03, -1.35403183e+01],
        #                                    [5.66974654e-03, 1.75704282e-03, 9.99982383e-01, 1.35447590e-01],
        #                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
        #                                   dtype=float)
        # transformation_matrix = np.linalg.inv(transformation_matrix)

        # Better initial transformation matrix: s110_lidar_valeo_north_west to s110_lidar_ouster_north
        # transformation_matrix = np.array([[0.96653359, 0.2558828, -0.01835234, 1.50034291],
        #                                   [-0.25498967, 0.96608398, 0.04076786, 14.49261505],
        #                                   [0.0281617, -0.03472385, 0.99900009, -0.68851967],
        #                                   [0.0, 0.0, 0.0, 1.0]], dtype=float)

        # print("transformation_matrix", repr(transformation_matrix))

    if transformation_matrix is not None:
        # continuous registration
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        continuous_registration_result = refine_registration(source_down, target_down, voxel_size,
                                                             transformation_matrix)
        if continuous_registration_result.inlier_rmse < inlier_rmse_best and continuous_registration_result.fitness > fitness_best:
            inlier_rmse_best = continuous_registration_result.inlier_rmse
            fitness_best = continuous_registration_result.fitness
            transformation_matrix = continuous_registration_result.transformation
            print("==========================")
            print("Better transformation matrix found (using continuous registration): \n",
                  repr(transformation_matrix))
            print("With frame index: %d" % idx_file)
            print("With better RMSE: %.4f" % inlier_rmse_best)
            print("With better fitness: %.4f" % fitness_best)
            print("==========================")

    # TODO: test evaluation
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target, 0.02, initial_transformation)
    # print(evaluation)
    return transformation_matrix, inlier_rmse_best, fitness_best


def get_initial_transformation_matrix_rtk_robosense():
    gps_lat_robosense = 11.6234528153807
    gps_long_robosense = 48.2509788374431
    gps_alt_robosense = 529.956770454546
    utm_east_robosense_initial, utm_north_robosense_initial, zone, _ = utm.from_latlon(gps_lat_robosense,
                                                                                       gps_long_robosense)
    gps_lat_rtk = 11.6234612182321
    gps_long_rtk = 48.250977342
    gps_alt_rtk = 529.820779761905
    utm_east_rtk_initial, utm_north_rtk_initial, zone, _ = utm.from_latlon(gps_lat_rtk, gps_long_rtk)
    translation_rtk_to_robosense_initial = np.array(
        [utm_east_robosense_initial - utm_east_rtk_initial,
         utm_north_robosense_initial - utm_north_rtk_initial,
         gps_alt_rtk - gps_alt_robosense], dtype=float)
    transformation_matrix_rtk_to_robosense = np.zeros((4, 4))
    transformation_matrix_rtk_to_robosense[0:3, 0:3] = np.identity(3)
    transformation_matrix_rtk_to_robosense[0:3, 3] = translation_rtk_to_robosense_initial
    transformation_matrix_rtk_to_robosense[3, 3] = 1.0
    print("transformation matrix rtk to robosense:")
    print(repr(transformation_matrix_rtk_to_robosense))
    return transformation_matrix_rtk_to_robosense


if __name__ == "__main__":
    rospy.init_node('lidar_point_cloud_registration_node')

    opt = parse_parameters()

    if bool(opt["publish_registered_point_clouds"]):
        point_cloud_registered_pub = rospy.Publisher(opt["ros_output_topic_registered_point_cloud"], PointCloud2,
                                                     queue_size=1)
        # for testing
        point_cloud_source_pub = rospy.Publisher("/point_cloud/source/points", PointCloud2,
                                                 queue_size=1)
        point_cloud_target_pub = rospy.Publisher("/point_cloud/target/points", PointCloud2,
                                                 queue_size=1)

    folder_path_point_cloud_source = opt["folder_path_point_cloud_source"]
    folder_path_point_cloud_target = opt["folder_path_point_cloud_target"]
    # folder_path_gps_data = opt["folder_path_gps_data"]
    # folder_path_imu_data = opt["folder_path_imu_data"]
    folder_path_rotation_data = opt["folder_path_rotation_data"]
    folder_path_translation_data = opt["folder_path_translation_data"]
    output_folder_path_registered_point_clouds = opt["output_folder_path_registered_point_clouds"]
    initial_voxel_size = float(opt["initial_voxel_size"])
    continuous_voxel_size = float(opt["continuous_voxel_size"])

    idx_file = 0
    # TODO: put parameter into launch file
    do_initial_registration = False
    registration_successfull = False
    transformation_matrix_best = None
    for point_cloud_source_filename, point_cloud_target_filename, rotation_filename, translation_filename in zip(
            sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target)),
            sorted(os.listdir(folder_path_rotation_data)), sorted(os.listdir(folder_path_translation_data))):
        print("processing point cloud with index: ", str(idx_file))
        if idx_file >=0 and idx_file <490 :
            point_cloud_source, header_source = read_point_cloud(
                os.path.join(folder_path_point_cloud_source, point_cloud_source_filename))
            point_cloud_target, header_target = read_point_cloud(
                os.path.join(folder_path_point_cloud_target, point_cloud_target_filename))

            tsl_vct = np.loadtxt(
                os.path.join(folder_path_translation_data, translation_filename),
                delimiter=",",
            )

            rot_mtx = R.from_quat(np.loadtxt(
                os.path.join(folder_path_rotation_data, rotation_filename),
                delimiter=",",
            ))

            # translation_vector_rotated = np.matmul(rot_mtx.as_matrix().T,
            #                                         tsl_vct)

            transformation_matrix_initial = np.zeros((4, 4))
            transformation_matrix_initial[0:3, 0:3] = rot_mtx.as_matrix()
            transformation_matrix_initial[0:3, 3] = tsl_vct
            transformation_matrix_initial[3, 3] = 1.0

            point_cloud_source = filter_point_cloud(point_cloud_source)
            point_cloud_target = filter_point_cloud(point_cloud_target)
            fitness_best = 0.000
            # TODO: temp test frame
            # if idx_file > 100:
            if idx_file >= 0:
                if registration_successfull and transformation_matrix_best is not None:
                    transformation_matrix_initial = transformation_matrix_best
                # transformation_matrix, inlier_rmse_best, fitness_best = register_point_clouds(do_initial_registration,
                #                                                                               point_cloud_source,
                #                                                                               point_cloud_target, idx_file,
                #                                                                               transformation_matrix_initial,
                #                                                                               initial_voxel_size,
                #                                                                               continuous_voxel_size)
                # TODO: temp:
                transformation_matrix = transformation_matrix_initial

                if fitness_best < 0.1:
                    transformation_matrix = transformation_matrix_initial
                else:
                    print("*********************************")
                    print("registration successfull")
                    print("*********************************")
                    print("frame idx: ", idx_file)
                    registration_successfull = True
                    transformation_matrix_best = transformation_matrix
            else:
                transformation_matrix = transformation_matrix_initial
            # TODO: keep matrix with best fitness
            # transformation_matrix = transformation_matrix_initial

            # transform point clouds
            if transformation_matrix is not None:
                # 1. transform point cloud from robosense lidar to gps/rtk device coordinate system
                one_column = np.ones((len(point_cloud_source), 1), dtype=float)
                point_cloud_source_homogeneous = np.concatenate((point_cloud_source[:, 0:3], one_column), axis=1)
                source_transformed = np.matmul(transformation_matrix, point_cloud_source_homogeneous.T).T

                source_transformed[:, 3] = point_cloud_source[:, 3]
            else:
                source_transformed = point_cloud_source

            # stack/merge points
            points_stacked = np.vstack(
                [source_transformed, point_cloud_target])

            point_cloud_source_filename = point_cloud_source_filename.split(".")[0]
            seconds = int(point_cloud_source_filename.split("_")[0])
            nano_seconds = int(point_cloud_source_filename.split("_")[1])

            # # Save transformed Pointclouds without mergging them

            # # # if bool(opt["save_transformed_point_clouds"]):
            # if bool(True):
            #     # save transformed source and target pointclouds after transformation
            #     file_name_source = str(seconds) + "_" + str(
            #         nano_seconds).zfill(
            #         9) + "_transformed_source_point_clouds.pcd"

            #     file_name_target = str(seconds) + "_" + str(
            #         nano_seconds).zfill(
            #         9) + "_transformed_target_point_clouds.pcd"

            #     write_point_cloud_with_intensities("/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/transformed" + "/" + file_name_source,
            #                                     source_transformed,
            #                                     header_source)
                
            #     write_point_cloud_with_intensities("/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/02_infrastructure_lidar_ouster/s110_lidar_ouster_south_driving_direction_east/transformed" + "/" + file_name_target,
            #                                     point_cloud_target,
            #                                     header_source)
                
            if bool(opt["save_registered_point_clouds"]):
                # save registered point clouds
                file_name = str("{0:.3f}".format(fitness_best)) + "_fitness_" + str(seconds) + "_" + str(
                    nano_seconds).zfill(
                    9) + "_registered_point_clouds.pcd"
                write_point_cloud_with_intensities(opt["output_folder_path_registered_point_clouds"] + "/" + file_name,
                                                points_stacked,
                                                header_source)
            if bool(opt["publish_registered_point_clouds"]):
                # publish registered point cloud
                point_cloud_header = Header()
                point_cloud_header.stamp.secs = seconds
                point_cloud_header.stamp.nsecs = nano_seconds
                point_cloud_header.frame_id = "registered"
                point_cloud_registered_msg = create_intensity_point_cloud2_msg(points_stacked, point_cloud_header)
                point_cloud_source_msg = create_intensity_point_cloud2_msg(source_transformed, point_cloud_header)
                point_cloud_target_msg = create_intensity_point_cloud2_msg(point_cloud_target, point_cloud_header)
                # while True:
                print("publishing point cloud...")
                point_cloud_registered_pub.publish(point_cloud_registered_msg)
                point_cloud_source_pub.publish(point_cloud_source_msg)
                point_cloud_target_pub.publish(point_cloud_target_msg)
                # time.sleep(0.1)

        idx_file = idx_file + 1

    print("==========================")
    print("Global results:")
    print("Best global RMSE: %.4f" % inlier_rmse_best)
    print("Best global fitness: %.4f" % fitness_best)
    print("Best global transformation_matrix: \n", repr(transformation_matrix))
    sys.exit()

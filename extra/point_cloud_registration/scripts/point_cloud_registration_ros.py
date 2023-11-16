import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import time

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from argparse import ArgumentParser
import open3d as o3d
import message_filters
from sensor_msgs.msg import PointCloud2
import threading

# initial_registration = True
initial_registration = 4
initial_inlier_rmse = 99999
initial_fitness = 0
initial_transformation = None

avg_callback_item = 1
avg_callback_time = 0
avg_init_item = 1
avg_init_time = 0
avg_init_rmse = 0
avg_init_fitness = 0
avg_con_item = 1
avg_con_time = 0
avg_con_rmse = 0
avg_con_fitness = 0
idx = 0


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


def continuous_registration(source, target, voxel_size, trans_init):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def create_point_cloud2(points, header):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )


def create_intensity_point_cloud2(points, header):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4),
        row_step=(itemsize * 4 * points.shape[0]),
        data=data
    )


def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float):
    '''get_xyzi_points
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']
    points[..., 3] = cloud_array['intensity']

    return points


def registration_callback(pointcloud_source, pointcloud_target):
    time_callback = time.time()
    global init_voxel_size
    global continuous_voxel_size
    global initial_registration
    global initial_transformation
    global initial_inlier_rmse
    global callback_lock
    global avg_callback_item
    global avg_callback_time
    global avg_init_item
    global avg_init_time
    global avg_init_rmse
    global avg_init_fitness
    global avg_con_item
    global avg_con_time
    global avg_con_rmse
    global avg_con_fitness
    global idx

    callback_lock.acquire()

    point_cloud_array_source = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_source)
    sorce_xyzi = get_xyzi_points(point_cloud_array_source, True)
    point_cloud_array_target = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_target)
    target_xyzi = get_xyzi_points(point_cloud_array_target, True)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(sorce_xyzi[:, 0:3])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_xyzi[:, 0:3])
    print("reading pcd time: ", time.time() - time_callback)

    trans = None
    voxel_size = continuous_voxel_size
    if initial_registration > 1:
        time_operation = time.time()
        voxel_size = init_voxel_size  # could be smaller to get a more accurate init trans
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_point_cloud(source, target,
                                                                                                 voxel_size)
        #result_ransac = execute_initial_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        #initial_transformation = result_ransac.transformation
        # TODO temporary hard code initial transformation matrix
        # initial transformation matrix ouster_north to ouster_south
        #initial_transformation = np.array([[9.68911602e-01, -2.47355442e-01, -5.05895822e-03, 2.07276299e+00],
        #                                   [2.47342195e-01, 9.68923216e-01, -3.10486184e-03, -1.35403183e+01],
        #                                   [5.66974654e-03, 1.75704282e-03, 9.99982383e-01, 1.35447590e-01],
        #                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
        #                                  dtype=float)
        # initial transformation matrix valeo_north_west to ouster_south
        initial_transformation = np.array([[0.75184951, 0.64600574, 0.13190484, -0.1206516],
                                            [-0.63314934, 0.76321442, -0.12894047, 1.24191049],
                                            [-0.18396796, 0.01342837, 0.98284051, -0.05209358],
                                            [0.0, 0.0, 0.0, 1.0]], dtype=float)

        print("initial_transformation", initial_transformation)
        result_refined = continuous_registration(source_down, target_down, voxel_size, initial_transformation)
        # source.transform(result_refined.transformation)
        if result_refined.inlier_rmse < initial_inlier_rmse and result_refined.fitness > initial_fitness:
            print("Better initial transformation found.")
            initial_transformation = result_refined.transformation
            # initial_registration = False
            print(initial_transformation)

        trans = result_refined.transformation

        initial_registration -= 1

        avg_init_item += 1
        avg_init_time += time.time() - time_operation
        avg_init_rmse += result_refined.inlier_rmse
        avg_init_fitness += result_refined.fitness
    else:
        time_operation = time.time()

        # time_read = time.time()

        # radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        result_refined = continuous_registration(source_down, target_down, voxel_size, initial_transformation)

        trans = result_refined.transformation
        print("continuous_transformation", trans)

        # # # evaluation
        # evaluation = o3d.pipelines.registration.evaluate_registration(
        #     source, target, 0.02, initial_transformation)
        # print(evaluation)

        avg_con_item += 1
        avg_con_time += time.time() - time_operation
        avg_con_rmse += result_refined.inlier_rmse
        print("inlier rmse:")
        print(result_refined.inlier_rmse)
        avg_con_fitness += result_refined.fitness

    # test the matrix:
    # transformation matrix ouster_north_ to ouster_south
    # trans = np.array([[9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e+00],
    #                   [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e+01],
    #                   [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
    #                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)

    if trans is not None:
        print("idx: ", str(idx))
        idx += 1
        print("trans: ", str(trans))
        cat = np.ones((len(sorce_xyzi), 1), dtype=np.float)
        sorce_cat = np.concatenate((sorce_xyzi[:, 0:3], cat), axis=1)
        sorce_transformed = np.matmul(sorce_cat, np.transpose(trans))
        sorce_transformed[:, 3] = sorce_xyzi[:, 3]

    else:
        sorce_transformed = sorce_xyzi
        print("ERROR: no transformation calculated")

    # time_post = time.time()

    pc_combined = np.concatenate((target_xyzi, sorce_transformed), axis=0)
    print(pc_combined.shape)

    ret_pc_header = Header()
    ret_pc_header.stamp = pointcloud_target.header.stamp
    ret_pc_header.frame_id = pointcloud_target.header.frame_id

    ret_points = np.float32(pc_combined)
    ret_points = pc_combined
    print("==========================")

    ret_pc = create_intensity_point_cloud2(ret_points, ret_pc_header)
    # print("post processing time: ", time.time() - time_post)

    # TODO: store as PCD
    fused_pc_pub.publish(ret_pc)

    if initial_registration <= 1:
        avg_callback_item += 1
        avg_callback_time += time.time() - time_callback
    print("callback time: %.4f s" % (time.time() - time_callback))
    avg_callback_item = max(avg_callback_item, 1)
    print("avg callback: %.4f s" % (avg_callback_time / avg_callback_item))
    print("avg init time: %.4f s" % (avg_init_time / avg_init_item), "rmse: %.4f" % (avg_init_rmse / avg_init_item),
          "fitness: %.4f" % (avg_init_fitness / avg_init_item))
    avg_con_item = max(avg_con_item, 1)
    print("avg con: %.4f s" % (avg_con_time / avg_con_item), "rmse: %.4f" % (avg_con_rmse / avg_con_item),
          "fitness: %.4f" % (avg_con_fitness / avg_con_item))
    callback_lock.release()


global init_voxel_size
global continuous_voxel_size
global callback_lock
if __name__ == "__main__":
    global init_voxel_size
    global continuous_voxel_size
    rospy.init_node('s110_lidar_point_cloud_registration_node')


    opt = {}
    try:
        # parse parameters from launch file
        opt["source_point_cloud"] = str(rospy.get_param('~source_point_cloud'))
        opt["target_point_cloud"] = str(rospy.get_param('~target_point_cloud'))
        opt["init_voxel_size"] = str(rospy.get_param('~init_voxel_size'))
        opt["continuous_voxel_size"] = str(rospy.get_param('~continuous_voxel_size'))
        opt["ros_output_topic_registered_point_cloud"] = str(rospy.get_param('~ros_output_topic_registered_point_cloud'))
    except:

        parser = ArgumentParser()
        parser.add_argument('--source_point_cloud', default='/s110/lidar/ouster/north/points',
                            help='topic of source point cloud')
        parser.add_argument('--target_point_cloud', default='/s110/lidar/ouster/south/points',
                            help='topic of target point cloud')
        parser.add_argument('--init_voxel_size', default=2, help='initial voxel size')
        parser.add_argument('--continuous_voxel_size', default=2, help='continuous voxel size')
        parser.add_argument('--ros_output_topic_registered_point_cloud', default="/s110/lidars/registered/points", help='ROS output topic of registered LiDAR point clouds.')
        opt = vars(parser.parse_args())

    source_sub = message_filters.Subscriber(opt["source_point_cloud"], PointCloud2)
    target_sub = message_filters.Subscriber(opt["target_point_cloud"], PointCloud2)
    init_voxel_size = float(opt["init_voxel_size"])
    continuous_voxel_size = float(opt["continuous_voxel_size"])

    callback_lock = threading.Lock()
    sync = message_filters.ApproximateTimeSynchronizer([source_sub, target_sub], 10, 0.1, allow_headerless=True)

    sync.registerCallback(registration_callback)

    fused_pc_pub = rospy.Publisher(opt["ros_output_topic_registered_point_cloud"], PointCloud2, queue_size=1)

    print("[+] Point cloud registration ros_node has started!")
    rospy.spin()

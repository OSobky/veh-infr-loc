import rospy
import ros_numpy
import numpy as np

np.set_printoptions(suppress=True)
import os
import time
import io
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from argparse import ArgumentParser
import open3d as o3d
import message_filters
from sensor_msgs.msg import PointCloud2
import threading

# initial_registration = True
initial_registration = 4
initial_inlier_rmse = 99999
initial_fitness = 0

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


def preprocess_point_cloud(point_cloud, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = point_cloud.voxel_down_sample(voxel_size)

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


def create_intensity_point_cloud2_msg(points, header):
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

    # remove rows (3D points) with all zero (xyz)
    points = points[~np.all(points[:, :3] == 0, axis=1)]

    # normalize intensities
    # valeo 0 - 10 to 0-255
    # ouster 1-2900 to 0-255
    # points = points[points[:, 3].argsort()]
    points[:, 3] = points[:, 3] / np.max(points[:, 3])
    return points


def write_point_cloud(output_file_path, point_cloud_registered_np):
    num_points = len(point_cloud_registered_np)
    header = "# .PCD v0.7 - Point Cloud Data file format\n" \
             + "VERSION 0.7\n" \
             + "FIELDS x y z intensity\n" \
             + "SIZE 4 4 4 4\n" \
             + "TYPE F F F F\n" \
             + "COUNT 1 1 1 1\n" \
             + "WIDTH " + str(num_points) + "\n" \
             + "HEIGHT 1\n" \
             + "VIEWPOINT 0 0 0 1 0 0 0\n" \
             + "POINTS " + str(num_points) + "\n" \
             + "DATA ascii\n"
    with open(output_file_path, "w") as writer:
        writer.write(header)
        np.savetxt(writer, point_cloud_registered_np, delimiter=" ", fmt="%.3f " * 4)


def registration_callback(point_cloud_target_msg, point_cloud_source_1_msg, point_cloud_source_2_msg):
    time_callback = time.time()
    global init_voxel_size
    global continuous_voxel_size
    global initial_registration
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
    global output_folder_path
    global idx
    print("entering callback: ", str(idx))
    idx += 1
    callback_lock.acquire()

    point_cloud_array_source_1 = ros_numpy.point_cloud2.pointcloud2_to_array(point_cloud_source_1_msg)
    source_1_xyzi = get_xyzi_points(point_cloud_array_source_1, True)
    point_cloud_array_source_2 = ros_numpy.point_cloud2.pointcloud2_to_array(point_cloud_source_2_msg)
    source_2_xyzi = get_xyzi_points(point_cloud_array_source_2, True)
    point_cloud_array_target = ros_numpy.point_cloud2.pointcloud2_to_array(point_cloud_target_msg)
    target_xyzi = get_xyzi_points(point_cloud_array_target, True)

    # point_cloud_source_1 = o3d.geometry.PointCloud()
    # point_cloud_source_1.points = o3d.utility.Vector3dVector(source_1_xyzi[:, 0:3])

    # point_cloud_source_2 = o3d.geometry.PointCloud()
    # point_cloud_source_2.points = o3d.utility.Vector3dVector(source_2_xyzi[:, 0:3])

    # point_cloud_target = o3d.geometry.PointCloud()
    # point_cloud_target.points = o3d.utility.Vector3dVector(target_xyzi[:, 0:3])
    # print("reading pcd time: ", time.time() - time_callback)

    # step: register source 1 with target
    # transformation_matrix_source_1_to_target = point_cloud_registration(continuous_voxel_size, init_voxel_size,
    #                                                                     initial_inlier_rmse, point_cloud_source_1,
    #                                                                     point_cloud_target, first_registration=True)
    # initial transformation matrix ouster_north to ouster_south
    # transformation_matrix_source_1_to_target = np.array(
    #     [[9.68911602e-01, -2.47355442e-01, -5.05895822e-03, 2.07276299e+00],
    #      [2.47342195e-01, 9.68923216e-01, -3.10486184e-03, -1.35403183e+01],
    #      [5.66974654e-03, 1.75704282e-03, 9.99982383e-01, 1.35447590e-01],
    #      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype=float)

    # final transformation matrix ouster_north to ouster_south
    transformation_matrix_source_1_to_target = np.array(
        [[9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e+00],
         [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e+01],
         [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)

    point_cloud_source_1_transformed = transform_point_cloud(source_1_xyzi, transformation_matrix_source_1_to_target)

    point_cloud_registered_np, point_cloud_registered_o3d = merge_point_clouds(point_cloud_source_1_transformed,
                                                                               target_xyzi)

    # step: register source 2 to with registered result of step 1 (source 1 + target)
    # transformation_matrix_source_2_to_target = point_cloud_registration(continuous_voxel_size, init_voxel_size,
    #                                                                     initial_inlier_rmse, point_cloud_source_2,
    #                                                                     point_cloud_registered_o3d,
    #                                                                     first_registration=False)
    # initial transformation matrix valeo_north_west to ouster_south
    transformation_matrix_source_2_to_target = np.array([[0.75184951, 0.64600574, 0.13190484, -0.1206516],
                                                         [-0.63314934, 0.76321442, -0.12894047, 1.24191049],
                                                         [-0.18396796, 0.01342837, 0.98284051, -0.05209358],
                                                         [0.0, 0.0, 0.0, 1.0]], dtype=float)
    point_cloud_source_2_transformed_np = transform_point_cloud(source_2_xyzi, transformation_matrix_source_2_to_target)

    point_cloud_registered_np, point_cloud_registered_o3d = merge_point_clouds(point_cloud_source_2_transformed_np,
                                                                               point_cloud_registered_np)

    print("==========================")
    # publish registered point cloud
    point_cloud_header = Header()
    point_cloud_header.stamp = point_cloud_target_msg.header.stamp
    point_cloud_header.frame_id = point_cloud_target_msg.header.frame_id
    point_cloud_msg = create_intensity_point_cloud2_msg(point_cloud_registered_np, point_cloud_header)
    # print("post processing time: ", time.time() - time_post)

    # store as PCD
    # NOTE: PointCloud2 timestamp is in milli seconds!
    secs = int(point_cloud_target_msg.header.stamp.secs)
    nsecs = int(point_cloud_target_msg.header.stamp.nsecs)
    # o3d.io.write_point_cloud(
    #     os.path.join(output_folder_path, str(secs) + "_" + str(nsecs).zfill(9) + ".pcd"),
    #     point_cloud_registered_o3d, write_ascii=True)
    write_point_cloud(os.path.join(output_folder_path, str(secs) + "_" + str(nsecs).zfill(9) + ".pcd"),
                      point_cloud_registered_np)
    registered_point_cloud_pub.publish(point_cloud_msg)

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


def merge_point_clouds(point_cloud_source_1_transformed, target_xyzi):
    point_cloud_registered_np = np.concatenate((target_xyzi, point_cloud_source_1_transformed), axis=0)
    # point_cloud_registered_np = np.float32(point_cloud_registered_np)
    point_cloud_registered_o3d = o3d.geometry.PointCloud()
    point_cloud_registered_o3d.points = o3d.utility.Vector3dVector(point_cloud_registered_np[:, 0:3])
    return point_cloud_registered_np, point_cloud_registered_o3d


def transform_point_cloud(source_1_xyzi, transformation_matrix_source_1_to_target):
    if transformation_matrix_source_1_to_target is not None:
        # replace intensity values with ones vector because they need not be transformed
        one_column = np.ones((len(source_1_xyzi), 1), dtype=np.float)
        point_cloud_source_1_homogeneous = np.concatenate((source_1_xyzi[:, 0:3], one_column), axis=1)
        point_cloud_source_1_transformed = np.matmul(point_cloud_source_1_homogeneous,
                                                     np.transpose(transformation_matrix_source_1_to_target))
        # recover intensity values
        point_cloud_source_1_transformed[:, 3] = source_1_xyzi[:, 3]
    else:
        point_cloud_source_1_transformed = source_1_xyzi
        print("\n")
        print('\x1b[0;30;41m' + 'ERROR: no transformation calculated' + '\x1b[0m')
        print("\n")
    return point_cloud_source_1_transformed


def point_cloud_registration(continuous_voxel_size, init_voxel_size, initial_inlier_rmse, point_cloud_source,
                             point_cloud_target, first_registration):
    global initial_registration, avg_init_item, avg_init_time, avg_init_rmse, avg_init_fitness, avg_con_item, avg_con_time, avg_con_rmse, avg_con_fitness, initial_transformation
    transformation_matrix = None
    voxel_size = continuous_voxel_size
    if initial_registration > 1:
        time_operation = time.time()
        voxel_size = init_voxel_size  # could be smaller to get a more accurate init trans
        point_cloud_source, point_cloud_target, point_cloud_source_down_sampled, point_cloud_target_down_sampled, source_fpfh, target_fpfh = prepare_point_cloud(
            point_cloud_source, point_cloud_target,
            voxel_size)
        result_ransac = execute_initial_registration(point_cloud_source_down_sampled, point_cloud_target_down_sampled,
                                                     source_fpfh, target_fpfh, voxel_size)
        initial_transformation = result_ransac.transformation

        print("initial_transformation", initial_transformation)
        result_refined = continuous_registration(point_cloud_source_down_sampled, point_cloud_target_down_sampled,
                                                 voxel_size, initial_transformation)
        # source.transform(result_refined.transformation)
        if result_refined.inlier_rmse < initial_inlier_rmse and result_refined.fitness > initial_fitness:
            print("Better initial transformation found.")
            initial_transformation = result_refined.transformation
            # initial_registration = False
            print(initial_transformation)

        transformation_matrix = result_refined.transformation

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
        point_cloud_source_down_sampled = point_cloud_source.voxel_down_sample(voxel_size)
        point_cloud_target_down_sampled = point_cloud_target.voxel_down_sample(voxel_size)

        if first_registration:
            initial_transformation = initial_transformation_source_1
        else:
            initial_transformation = initial_transformation_source_2

        result_refined = continuous_registration(point_cloud_source_down_sampled, point_cloud_target_down_sampled,
                                                 voxel_size, initial_transformation)
        if result_refined.inlier_rmse < initial_inlier_rmse and result_refined.fitness > initial_fitness:
            initial_transformation = result_refined.transformation

        transformation_matrix = result_refined.transformation
        print("continuous_transformation", transformation_matrix)

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
    return transformation_matrix


global init_voxel_size
global continuous_voxel_size
global callback_lock

if __name__ == "__main__":
    global init_voxel_size
    global continuous_voxel_size
    global output_folder_path
    global initial_transformation_source_1
    global initial_transformation_source_2
    rospy.init_node('pc_registration_node')

    parser = ArgumentParser()
    parser.add_argument('--source_point_cloud1', default='/s110/lidar/ouster/north/points',
                        help='topic of source point cloud 1 (will be transformed to target point cloud frame)')
    parser.add_argument('--source_point_cloud2', default='/s110/lidar/valeo/north_west/points',
                        help='topic of source point cloud 2 (will be transformed to target point cloud frame)')
    parser.add_argument('--target_point_cloud', default='/s110/lidar/ouster/south/points',
                        help='topic of target point cloud (remains static and will not be transformed)')
    parser.add_argument('--init_voxel_size', default=2, help='initial voxel size')
    parser.add_argument('--continuous_voxel_size', default=2, help='continuous voxel size')
    parser.add_argument('--output_folder_path', default="output",
                        help='Output folder path to store registered .pcd files.')
    parser.add_argument('--ros_topic_point_cloud_registered', default="/s110/lidars/registered/points",
                        help='Topic that is published for registered lidar point clouds.')
    args = parser.parse_args()

    registered_point_cloud_pub = rospy.Publisher(args.ros_topic_point_cloud_registered, PointCloud2, queue_size=1)

    source1_sub = message_filters.Subscriber(args.source_point_cloud1, PointCloud2)
    source2_sub = message_filters.Subscriber(args.source_point_cloud2, PointCloud2)
    target_sub = message_filters.Subscriber(args.target_point_cloud, PointCloud2)
    init_voxel_size = float(args.init_voxel_size)
    continuous_voxel_size = float(args.continuous_voxel_size)
    output_folder_path = args.output_folder_path

    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    callback_lock = threading.Lock()
    sync = message_filters.ApproximateTimeSynchronizer([target_sub, source1_sub, source2_sub], 500, 15,
                                                       allow_headerless=True)
    sync.registerCallback(registration_callback)

    print("[+] Point cloud registration ros_node has started!")
    rospy.spin()

import json
import os
import sys
import time

import numpy as np

from transform_and_merge import get_initial_transformation_matrix

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


def read_point_cloud(input_file_path):
    lines = None
    with open(input_file_path, "r") as reader:
        lines = reader.readlines()
    header = lines[:11]
    point_cloud_array = np.array(pd.read_csv(input_file_path, sep=' ', skiprows=11, dtype=float).values)[:, :4]
    return point_cloud_array, header


def write_point_cloud_with_intensities(output_file_path, point_cloud_array, header):
    # update num points
    header[2] = "FIELDS x y z intensity rgb\n"
    header[3] = "SIZE 4 4 4 4 4\n"
    header[4] = "TYPE F F F F F\n"
    header[5] = "COUNT 1 1 1 1 1\n"
    header[6] = "WIDTH " + str(len(point_cloud_array)) + "\n"
    header[7] = "HEIGHT 1" + "\n"
    header[9] = "POINTS " + str(len(point_cloud_array)) + "\n"
    with open(output_file_path, 'w') as writer:
        for header_line in header:
            writer.write(header_line)
    df = pd.DataFrame(point_cloud_array)
    df.to_csv(output_file_path, sep=" ", header=False, mode='a', index=False)


def filter_point_cloud(point_cloud):
    # normalize intensities
    point_cloud[:, 3] *= (1 / point_cloud[:, 3].max())

    # remove zero rows
    point_cloud = point_cloud[~np.all(point_cloud[:, :3] == 0.0, axis=1)]

    # remove nans
    point_cloud = point_cloud[~np.isnan(point_cloud).any(axis=1), :]

    # remove points above 100 m distance
    # distances = [(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]) ^ 0.5 for point in point_cloud]
    distances = np.array([np.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]) for point in point_cloud])
    point_cloud = point_cloud[distances < 150.0, :]
    return point_cloud


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


def parse_parameters():
    opt = {}
    try:
        # parse parameters from launch file
        # TODO: add launch file
        opt["source_point_cloud"] = str(rospy.get_param('~source_point_cloud'))
        opt["target_point_cloud"] = str(rospy.get_param('~target_point_cloud'))

        opt["folder_path_gps_data"] = str(rospy.get_param('~folder_path_gps_data'))
        opt["folder_path_imu_data"] = str(rospy.get_param('~folder_path_imu_data'))

        opt["folder_path_rotation_data"] = str(rospy.get_param('~folder_path_rotation_data'))
        opt["folder_path_translation_data"] = str(rospy.get_param('~folder_path_translation_data'))

        opt["initial_voxel_size"] = str(rospy.get_param('~initial_voxel_size'))
        opt["continuous_voxel_size"] = str(rospy.get_param('~continuous_voxel_size'))

        opt["save_registered_point_clouds"] = str(rospy.get_param('~save_registered_point_clouds'))
        opt["output_folder_path_registered_point_clouds"] = str(
            rospy.get_param('~output_folder_path_registered_point_clouds'))

        opt["publish_registered_point_clouds"] = bool(
            rospy.get_param('~publish_registered_point_clouds'))
        opt["ros_output_topic_registered_point_cloud"] = str(
            rospy.get_param('~ros_output_topic_registered_point_cloud'))
    except:

        parser = ArgumentParser()

        parser.add_argument('--folder_path_point_cloud_source',
                            default='/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_valeo_north_west/',
                            help='folder path of source point cloud (will be transformed to target point cloud frame)')
        parser.add_argument('--folder_path_point_cloud_target',
                            default='/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_south/',
                            help='folder path of target point cloud (remains static and will not be transformed)')

        parser.add_argument('--folder_path_gps_data',
                            default='input/gps',
                            help='folder path of gps data')

        parser.add_argument('--folder_path_imu_data',
                            default='input/imu',
                            help='folder path of imu data')

        parser.add_argument('--folder_path_rotation_data',
                            default='input/rotation',
                            help='folder path of rotation data')

        parser.add_argument('--folder_path_translation_data',
                            default='input/translation',
                            help='folder path of translation data')

        parser.add_argument('--initial_voxel_size', default=2, help='initial voxel size')
        parser.add_argument('--continuous_voxel_size', default=2, help='continuous voxel size')

        parser.add_argument('--save_registered_point_clouds', action='store_true',
                            help='Save registered point cloud (By default it is not saved)')
        parser.add_argument('--output_folder_path_registered_point_clouds', default="output",
                            help='Output folder path to save registered point clouds (default: output)')

        parser.add_argument('--publish_registered_point_clouds', action="store_true",
                            help='Publish registered point clouds (By default point clouds are not published)')
        parser.add_argument('--ros_output_topic_registered_point_cloud', default="output",
                            help='ROS output publish topic to publish registered point clouds (default: /s110/lidars/registered/points)')
        # TODO: add flag --view_registered_point_clouds

        opt = vars(parser.parse_args())
    return opt

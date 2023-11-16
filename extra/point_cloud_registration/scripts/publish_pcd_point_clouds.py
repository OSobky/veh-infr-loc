import argparse
import os
import time

import pandas as pd
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


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


def publish_pcd_point_cloud(point_cloud_registered_pub, point_cloud_array):
    point_cloud_header = Header()
    point_cloud_header.stamp.secs = 0
    point_cloud_header.stamp.nsecs = 0
    point_cloud_header.frame_id = "registered"
    point_cloud_registered_msg = create_intensity_point_cloud2_msg(point_cloud_array, point_cloud_header)
    point_cloud_registered_pub.publish(point_cloud_registered_msg)


def read_point_cloud(input_file_path):
    lines = None
    with open(input_file_path, "r") as reader:
        lines = reader.readlines()
    header = lines[:11]
    point_cloud_array = np.array(pd.read_csv(input_file_path, sep=' ', skiprows=11, dtype=float).values)[:, :4]
    return point_cloud_array, header


if __name__ == '__main__':
    rospy.init_node("publisher_node")
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument('--input_folder_path_point_clouds', default='input/images',
                           help='Input source path of e.g. image .png frames, LiDAR .pcd frames or detections .json files (default: input/images)')
    args = argparser.parse_args()
    point_cloud_registered_pub = rospy.Publisher("/registered/points", PointCloud2,
                                                 queue_size=1)
    for point_cloud_file_name in sorted(os.listdir(args.input_folder_path_point_clouds)):
        point_cloud_array, header = read_point_cloud(
            os.path.join(args.input_folder_path_point_clouds, point_cloud_file_name))
        publish_pcd_point_cloud(point_cloud_registered_pub, point_cloud_array)
        time.sleep(0.1)

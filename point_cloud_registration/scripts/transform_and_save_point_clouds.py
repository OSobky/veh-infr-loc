import rospy
from argparse import ArgumentParser
import numpy as np

np.set_printoptions(suppress=True)
import os
import pandas as pd


# Merge the point clouds using the optimal transformation matrix that was obtained by (point_cloud_registration_pcd.py using all input .pcd files) -> optimal result
# NOTE: use the final transformation matrix obtained by merge_point_clouds.py

# Example: python merge_point_clouds.py --folder_path_point_cloud_source1 /mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_south/ --folder_path_point_cloud_source2 /mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_valeo_north_west/ --folder_path_point_cloud_target /mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_north/ --output_folder_path /mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds_registered/s110_lidar_ouster_north/

def write_point_cloud(output_file_path, point_cloud_array_final):
    num_points = len(point_cloud_array_final)
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
        np.savetxt(writer, point_cloud_array_final, delimiter=" ", fmt="%.3f " * 4)


def read_point_cloud(input_file_path):
    return np.array(pd.read_csv(input_file_path, sep=' ', skiprows=11, dtype=float).values)[:, :4]


def merge_point_clouds(point_cloud_array1, point_cloud_array2):
    return np.concatenate((point_cloud_array1, point_cloud_array2), axis=0)


def normalize_intensities(point_cloud_array):
    # normalize intensities
    # valeo 0 - 10 to 0-1.0
    # ouster 1-3239.0 to 0-1.0
    point_cloud_array[:, 3] = point_cloud_array[:, 3] / np.max(point_cloud_array[:, 3])
    return point_cloud_array


def transform_point_cloud(point_cloud_array, transformation_matrix):
    # replace intensity values with ones vector because they need not be transformed
    one_column = np.ones((len(point_cloud_array), 1), dtype=float)
    point_cloud_array_homogeneous = np.concatenate((point_cloud_array[:, 0:3], one_column), axis=1)
    point_cloud_array_transformed = np.matmul(point_cloud_array_homogeneous,
                                              np.transpose(transformation_matrix))

    # recover intensity values
    point_cloud_array_transformed[:, 3] = point_cloud_array[:, 3]
    return point_cloud_array_transformed


if __name__ == "__main__":
    rospy.init_node('point_cloud_transformation')

    parser = ArgumentParser()
    parser.add_argument('--folder_path_point_cloud_source',
                        type=str,
                        default="/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds_registered/s110_lidar_ouster_south",
                        help='folder path of source point cloud (will be transformed to target point cloud frame)')
    parser.add_argument('--folder_path_point_cloud_target',
                        type=str,
                        default="/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_north",
                        help='folder path of target point cloud (to grab file names)')
    parser.add_argument('--output_folder_path',
                        default="/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds_registered/s110_lidar_ouster_north",
                        help='Output folder path to store transformed .pcd files.')
    args = parser.parse_args()

    folder_path_point_cloud_source = args.folder_path_point_cloud_source
    folder_path_point_cloud_target = args.folder_path_point_cloud_target
    output_folder_path = args.output_folder_path

    # final transformation matrix (s110_lidar_ouster_south to s110_lidar_ouster_north)
    transformation_matrix_s110_lidar_ouster_south_to_s110_lidar_ouster_north = np.array(
        [[9.58976475e-01, 2.83448899e-01, 4.56533431e-03, 2.55136126e+00],
         [-2.83475533e-01, 9.58953986e-01, 6.99099595e-03, 1.37223201e+01],
         [-2.39635544e-03, - 7.99836123e-03, 9.99965141e-01, -4.18877942e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)

    for point_cloud_source_filename, point_cloud_target_filename in zip(
            sorted(os.listdir(folder_path_point_cloud_source)), sorted(os.listdir(folder_path_point_cloud_target))):
        point_cloud_source = read_point_cloud(
            os.path.join(folder_path_point_cloud_source, point_cloud_source_filename))
        point_cloud_array_source = np.array(point_cloud_source)

        point_cloud_array_source_transformed = transform_point_cloud(point_cloud_array_source,
                                                                     transformation_matrix_s110_lidar_ouster_south_to_s110_lidar_ouster_north)

        write_point_cloud(
            os.path.join(output_folder_path, point_cloud_target_filename),
            point_cloud_array_source_transformed)

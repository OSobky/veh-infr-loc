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
    one_column = np.ones((len(point_cloud_array), 1), dtype=np.float)
    point_cloud_array_homogeneous = np.concatenate((point_cloud_array[:, 0:3], one_column), axis=1)
    point_cloud_array_transformed = np.matmul(point_cloud_array_homogeneous,
                                              np.transpose(transformation_matrix))

    # recover intensity values
    point_cloud_array_transformed[:, 3] = point_cloud_array[:, 3]
    return point_cloud_array_transformed


if __name__ == "__main__":
    rospy.init_node('point_cloud_fusion')

    parser = ArgumentParser()
    parser.add_argument('--folder_path_point_cloud_source1',
                        type=str,
                        help='folder path of first source point cloud (will be transformed to target point cloud frame)')
    parser.add_argument('--folder_path_point_cloud_source2',
                        type=str,
                        help='folder path of second source point cloud (will be transformed to target point cloud frame)')
    parser.add_argument('--folder_path_point_cloud_target',
                        default='/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds/s110_lidar_ouster_south/',
                        help='folder path of target point cloud (remains static and will not be transformed)')
    parser.add_argument('--output_folder_path',
                        default="/mnt/hdd_data1/28_datasets/00_a9_dataset/01_R1_sequences/04_R1_S4/04_point_clouds_registered/",
                        help='Output folder path to store registered .pcd files.')
    args = parser.parse_args()

    folder_path_point_cloud_source1 = args.folder_path_point_cloud_source1
    if args.folder_path_point_cloud_source2 is not None:
        folder_path_point_cloud_source2 = args.folder_path_point_cloud_source2
    else:
        folder_path_point_cloud_source2 = None
    folder_path_point_cloud_target = args.folder_path_point_cloud_target
    output_folder_path = args.output_folder_path

    # final transformation matrix (s110_lidar_ouster_north to s110_lidar_ouster_south)
    # transformation_matrix_s110_lidar_ouster_north_to_s110_lidar_ouster_south = np.array(
    #     [[9.58895265e-01, -2.83760227e-01, -6.58645965e-05, 1.41849928e+00],
    #      [2.83753514e-01, 9.58874128e-01, -6.65957109e-03, -1.37385689e+01],
    #      [1.95287726e-03, 6.36714187e-03, 9.99977822e-01, 3.87637894e-01],
    #      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)
    # final transformation matrix (s110_lidar_ouster_south to s110_lidar_ouster_north)
    transformation_matrix_s110_lidar_ouster_south_to_s110_lidar_ouster_north = np.array(
        [[9.58976475e-01, 2.83448899e-01, 4.56533431e-03, 2.55136126e+00],
         [-2.83475533e-01, 9.58953986e-01, 6.99099595e-03, 1.37223201e+01],
         [-2.39635544e-03, - 7.99836123e-03, 9.99965141e-01, -4.18877942e-01],
         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=float)

    # final transformation matrix (s110_lidar_valeo_north_west to s110_lidar_ouster_south)
    transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_south = np.array(
        [[0.75423894, 0.64340394, 0.13097705, -0.21929972],
         [-0.63043731, 0.76537945, -0.12939507, 1.07040641],
         [-0.18350044, 0.01502199, 0.98290484, -0.04990497],
         [0.0, 0.0, 0.0, 1.0]], dtype=float)

    # final transformation matrix (s110_lidar_valeo_north_west to s110_lidar_ouster_north)
    # does not work
    # transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_north = np.array(
    #     [[0.96653359, 0.2558828, -0.01835234, 1.50034291],
    #      [-0.25498967, 0.96608398, 0.04076786, 14.49261505],
    #      [0.0281617, -0.03472385, 0.99900009, -0.68851967],
    #      [0., 0., 0., 1.]], dtype=float)
    transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_north = np.matmul(
        transformation_matrix_s110_lidar_ouster_south_to_s110_lidar_ouster_north,
        transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_south)
    print(repr(transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_north))

    if folder_path_point_cloud_source2 is not None:
        lists = zip(sorted(os.listdir(folder_path_point_cloud_source1)),
                    sorted(os.listdir(folder_path_point_cloud_source2)),
                    sorted(os.listdir(folder_path_point_cloud_target)))
    else:
        lists = zip(sorted(os.listdir(folder_path_point_cloud_source1)),
                    sorted(os.listdir(folder_path_point_cloud_source1)),
                    sorted(os.listdir(folder_path_point_cloud_target)))

    for point_cloud_source1_filename, point_cloud_source2_filename, point_cloud_target_filename in lists:
        point_cloud_source1 = read_point_cloud(
            os.path.join(folder_path_point_cloud_source1, point_cloud_source1_filename))
        point_cloud_array_source1 = np.array(point_cloud_source1)
        point_cloud_array_source1 = normalize_intensities(point_cloud_array_source1)

        if folder_path_point_cloud_source2 is not None:
            point_cloud_source2 = read_point_cloud(
                os.path.join(folder_path_point_cloud_source2, point_cloud_source2_filename))
            point_cloud_array_source2 = np.array(point_cloud_source2)
            point_cloud_array_source2 = normalize_intensities(point_cloud_array_source2)

        point_cloud_target = read_point_cloud(
            os.path.join(folder_path_point_cloud_target, point_cloud_target_filename))
        point_cloud_array_target = np.array(point_cloud_target)
        point_cloud_array_target = normalize_intensities(point_cloud_array_target)

        point_cloud_array_source_1_transformed = transform_point_cloud(point_cloud_array_source1,
                                                                       transformation_matrix_s110_lidar_ouster_south_to_s110_lidar_ouster_north)

        if folder_path_point_cloud_source2 is not None:
            point_cloud_array_source_2_transformed = transform_point_cloud(point_cloud_array_source2,
                                                                           transformation_matrix_s110_lidar_valeo_north_west_to_s110_lidar_ouster_north)

        point_cloud_array_target = merge_point_clouds(point_cloud_array_source_1_transformed,
                                                      point_cloud_array_target)
        if folder_path_point_cloud_source2 is not None:
            point_cloud_array_final = merge_point_clouds(point_cloud_array_source_2_transformed,
                                                         point_cloud_array_target)
        write_point_cloud(os.path.join(output_folder_path, point_cloud_target_filename),
                          point_cloud_array_target)

import copy
import os
import numpy as np
import open3d as o3d

import json
import utm

from scipy.spatial.transform import Rotation as R


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
# For Global registeration
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# downsample the the dataset for global regsiteration
def prepare_dataset(src_pcd_path, trgt_pcd_path, voxel_size, vis=False):
    print(":: Load two point clouds and disturb initial pose.")


    source = o3d.io.read_point_cloud(src_pcd_path, remove_nan_points=True, remove_infinite_points=True)
    target = o3d.io.read_point_cloud(trgt_pcd_path, remove_nan_points=True, remove_infinite_points=True)

    print(": Clean point clouds remove outliers and NaN, -0 values.")
    
    # # Clean pcd: remove outliers and NaN, -0 values This is needed anymore since the data is already cleaned
    # source = clean_pcd(source, viz=False)
    # target = clean_pcd(target, viz=False)

    # trans_init = np.asarray([[1, 0, 0, 0],
    #                      [0, 1, 0, 0],
    #                      [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])

    # trans_init = initial_tf_from_gps("", "")

    # source.transform(trans_init)
    if vis:
        draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# execute the global registeration 
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
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


# Create a function to read GPS data and generate initial transformation matrix
def initial_tf_from_gps(gps_path, imu_path):

    float_formatter = "{:.8f}".format
    np.set_printoptions(suppress=True, formatter={'float_kind': float_formatter})

    gps_json = json.load(open(gps_path))
    imu_json = json.load(open(imu_path))
    lat, long, alt = float(gps_json["lat"]), float(gps_json["long"]), float(gps_json["alt"])

    quaternion_x ,quaternion_y ,quaternion_z ,quaternion_w =  float(imu_json["quaternion_x"]), float(imu_json["quaternion_y"]), float(imu_json["quaternion_z"]), float(imu_json["quaternion_w"])


    # # 1. transformation matrix (vehicle lidar to infrastructure lidar)
    # # 1.1 vehicle lidar pose
    # utm_east_vehicle_lidar, utm_north_vehicle_lidar, zone, _ = utm.from_latlon(lat, long)
    # utm_east_vehicle_lidar = utm_east_vehicle_lidar
    # utm_north_vehicle_lidar = utm_north_vehicle_lidar
    # altitude_vehicle_lidar = alt

    # rotation_x_vehicle_lidar = quaternion_x
    # rotation_y_vehicle_lidar = quaternion_y
    # rotation_z_vehicle_lidar = quaternion_z
    # rotation_w_vehicle_lidar = quaternion_w
    # rotation_roll_vehicle_lidar, rotation_pitch_vehicle_lidar, rotation_yaw_vehicle_lidar = R.from_quat(
    #     [rotation_x_vehicle_lidar, rotation_y_vehicle_lidar, rotation_z_vehicle_lidar,
    #      rotation_w_vehicle_lidar]).as_euler('xyz', degrees=True)
    


    # # 1.2 infrastructure lidar pose
    # utm_east_s110_lidar_ouster_south = 695308.460000000 - 0.5
    # utm_north_s110_lidar_ouster_south = 5347360.569000000 + 2.5
    # altitude_s110_lidar_ouster_south = 534.3500000000000 + 1.0

    # rotation_roll_s110_lidar_ouster_south = 0  # 1.79097398157454 # pitch
    # rotation_pitch_s110_lidar_ouster_south = 1.1729642881072  # roll
    # rotation_yaw_s110_lidar_ouster_south = 172  # 172.693672075377
    

    # translation_vehicle_lidar_to_s110_lidar_ouster_south = np.array(
    #     [utm_east_s110_lidar_ouster_south - utm_east_vehicle_lidar,
    #      utm_north_s110_lidar_ouster_south - utm_north_vehicle_lidar,
    #      altitude_s110_lidar_ouster_south - altitude_vehicle_lidar], dtype=float)
    # print("translation vehicle lidar to s110_lidar_ouster_south: ",
    #       translation_vehicle_lidar_to_s110_lidar_ouster_south)
    
    # rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south = R.from_rotvec(
    #     [rotation_roll_s110_lidar_ouster_south - rotation_roll_vehicle_lidar,
    #      rotation_pitch_s110_lidar_ouster_south - rotation_pitch_vehicle_lidar,
    #      rotation_yaw_s110_lidar_ouster_south - rotation_yaw_vehicle_lidar],
    #     degrees=True).as_matrix().T
        
    # transformation_matrix = np.identity(4)
    # # transformation_matrix[0, 0] = -1
    # transformation_matrix[0:3, 0:3] = rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south
    # transformation_matrix[0, 0] = -1 * transformation_matrix[0, 0]
    # transformation_matrix[0:3, 3] = translation_vehicle_lidar_to_s110_lidar_ouster_south

    # print("transformation matrix: vehicle_lidar to s110_lidar_ouster_south")
    # print(repr(transformation_matrix))

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

    # print("euler angles (lidar vehicle): ",
    #       [rotation_roll_vehicle_lidar, rotation_pitch_vehicle_lidar, rotation_yaw_vehicle_lidar])
    # rotation_yaw_vehicle_lidar = -rotation_yaw_vehicle_lidar

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
    # print("translation vehicle lidar to s110_lidar_ouster_south: ",
    #       translation_vehicle_lidar_to_s110_lidar_ouster_south)

    rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south = R.from_rotvec(
        [rotation_roll_s110_lidar_ouster_south - rotation_roll_vehicle_lidar,
         rotation_pitch_s110_lidar_ouster_south - rotation_pitch_vehicle_lidar,
         rotation_yaw_s110_lidar_ouster_south - rotation_yaw_vehicle_lidar],
        degrees=True).as_matrix().T

    # print("rotation yaw final: ", str(rotation_yaw_s110_lidar_ouster_south - rotation_yaw_vehicle_lidar))

    translation_vector_rotated = np.matmul(rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south,
                                           translation_vehicle_lidar_to_s110_lidar_ouster_south)
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_matrix_vehicle_lidar_to_s110_lidar_ouster_south
    transformation_matrix[0:3, 3] = -translation_vector_rotated
    transformation_matrix[3, 3] = 1.0
    # print("transformation matrix: vehicle_lidar to s110_lidar_ouster_south")
    # print(repr(transformation_matrix))

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

    # print("transformation matrix: vehicle_lidar to s110_lidar_ouster_south with blender matrix")
    # print(repr(transformation_matrix))

    return transformation_matrix



def matrix_to_kitti_format(matrix):
    """
    matrix is 4x4. We need to get rid of the last row
    """

    # remove last row
    matrix = matrix[:-1][:]
    # Flatten the matrix in row-major order
    flattened_matrix = matrix.flatten('C')
    
    # Convert to string with space-separated values
    kitti_format = ' '.join(map(str, flattened_matrix))
    
    return kitti_format

# Example usage:

# Create a 4x4 transformation matrix
transformation_matrix = np.eye(4)

# Convert to KITTI format
kitti_format = matrix_to_kitti_format(transformation_matrix)

print(kitti_format)



"""

BELOW is CREATED METHODS WHICH IS NOT NEEDED ANY MORE!!!

"""


def clean_pcd(pcd, nb_neighbors=20, std_ratio=2.0, viz=False):
    """
    Clean Point clouds from NAN & outliers points

    steps: TODO
    """

    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if viz:
        _display_inlier_outlier(pcd, ind)
    
    return cl

def _display_inlier_outlier(cloud, ind):
    """
    Display inliers and outliers in a point cloud


    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    

def clean_save_pcd(input_dir, output_dir=""):
    """
    Clean and save the data to be loaded and used right away

    Input:
    - input_dir: path to the directory containing the PCDs  
    - output_dir: path to save cleaned data
    """

    if output_dir=="":
        output_dir = os.path.join(input_dir, "../clean/")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)

    print(f":: Staring the cleaning process. The output pcd would be in this folder: {output_dir}")
    for input_pcd_filename in sorted(os.listdir(input_dir)):

        print(f"input pcd filename to be cleaned: {input_pcd_filename}")
        input_pcd = o3d.io.read_point_cloud(os.path.join(input_dir,input_pcd_filename))
        clean_pcd(input_pcd, viz=False)
        o3d.io.write_point_cloud(output_dir + input_pcd_filename, input_pcd)
        print(f":: Cleaned pcd saved to: {output_dir + input_pcd_filename}")
             
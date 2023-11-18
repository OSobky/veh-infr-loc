import copy
import os
from datetime import datetime
from time import sleep
import numpy as np
import open3d as o3d

import json
import utm

from scipy.spatial.transform import Rotation as R


def create_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Infa-Veh-Reg Timestamp: ', width=1920, height=1080)
    ctr = vis.get_view_control()
    rndr = vis.get_render_option()
    param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    rndr.load_from_json("RenderOption.json")

    return vis

def stream_registeration_results(source, target, transformation, exp_path ,index, vis):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])

   

    # Update the visualizer
    vis.clear_geometries()
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.poll_events()
    vis.update_renderer()
    


def draw_registration_result(source, target, transformation, exp_path ,index):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Infa-Veh-Reg Timestamp: {}'.format(index), width=1920, height=1080)
    ctr = vis.get_view_control()
    rndr = vis.get_render_option()
    param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")

    for geometry in [source_temp, target_temp]:
        vis.add_geometry(geometry)

    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    rndr.load_from_json("RenderOption.json")

    # Run the visualizer
    # vis.run()
    vis.capture_screen_image(os.path.join(exp_path, f"viewpoint_{index}.png"), True)  # Optional, to save a screenshot  
    vis.destroy_window()

def save_draw_registration_result(source, target, transformation, exp_path ,index):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Infa-Veh-Reg Timestamp: {}'.format(index), width=1920, height=1080)
    for geometry in [source_temp, target_temp]:
        vis.add_geometry(geometry)
    
    # Run the visualizer
    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)    
    vis.destroy_window()



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

    # Add the transformation matrix from RTK to Robosense
    transformation_matrix_rtk_to_robosense = get_initial_transformation_matrix_rtk_robosense()

    return transformation_matrix @ transformation_matrix_rtk_to_robosense

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
    # print("transformation matrix rtk to robosense:")
    # print(repr(transformation_matrix_rtk_to_robosense))
    return transformation_matrix_rtk_to_robosense

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


def calculate_rmse(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))


def evo(gt_path, est_path):
    from evo.core import metrics
    from evo.tools import plot
    from evo.tools import file_interface
    from evo.core import sync

    # Load trajectories from files (in TUM, KITTI, EuRoC, ROS bag, ... format)
    traj_ref = file_interface.read_kitti_poses_file(gt_path)
    traj_est = file_interface.read_kitti_poses_file(est_path)

    # Synchronize the trajectories (if they're not aligned)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # Calculate APE
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    print('APE:', ape_stats)

    # Calculate RPE
    rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0, delta_unit=metrics.Unit.frames)
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_stats = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
    print('RPE:', rpe_stats)

    # Plotting the result (optional)
    plot_mode = metrics.PlotMode.xy
    plot_collection = plot.PlotCollection("Example")
    # Trajectories plot
    plot_collection.add_figure().add_plot(metrics.PlotTrajectories([traj_ref, traj_est], plot_mode))
    # APE and RPE plots
    plot_collection.add_figure().add_plot(metrics.PlotMetric(ape_metric, plot_mode))
    plot_collection.add_figure().add_plot(metrics.PlotMetric(rpe_metric, plot_mode))
    plot_collection.show()

def extract_iterations(log_output):
    lines = log_output.strip().split("\n")
    for line in reversed(lines):
        if "ICP Iteration" in line:
            iteration_number = int(line.split("#")[1].split(":")[0])
            return iteration_number + 1
    return 0

def create_experiment_folder(parent_dir, exp_name):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a path to the new folder inside the "experiments" directory
    path = os.path.join(parent_dir, timestamp + "_" + exp_name)

    # Create the experiment folder
    os.mkdir(path)

    # Return the path to the new folder
    return path


def set_view_parameters(vis, params_json_path):

    # Read the JSON file
    with open(params_json_path, 'r') as file:
        params_json = json.load(file)

    # Extract the first trajectory object
    trajectory = params_json['trajectory'][0]

    # Retrieve the view control object
    view_control = vis.get_view_control()

    # Set the parameters based on the JSON object
    if 'zoom' in trajectory:
        view_control.set_zoom(trajectory['zoom'])
    if 'front' in trajectory:
        view_control.set_front(np.array(trajectory['front']))
    if 'lookat' in trajectory:
        view_control.set_lookat(np.array(trajectory['lookat']))
    if 'up' in trajectory:
        view_control.set_up(np.array(trajectory['up']))
    if 'field_of_view' in trajectory:
        current_fov = view_control.get_field_of_view()
        delta_fov = trajectory['field_of_view'] - current_fov
        view_control.change_field_of_view(delta_fov)



def remove_ground(point_cloud, ground_height=0):
    column = np.asarray(point_cloud.points)[:,2]
    filtered_frame = np.asarray(point_cloud.points)[column > ground_height]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_frame)
    return pcd

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
             
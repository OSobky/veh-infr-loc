import os


# Define a function to compare the first 7 lines of two files
def modify_files(file1, file2):
    # Open the two files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Read the first 7 lines of each file
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
        # print("lines 6 & 7 of file one", f1_lines)
        # print("lines 6 & 7 of file two", f2_lines)

        # Modify the lines if necessary
        # For example, you could replace the first line of file1 with the first line of file2
        f2_lines[0:8] = f1_lines[0:8]
        # Write the modified lines back to file1
        with open(file2, 'w') as f:
            f.writelines(f2_lines)
        print("Modified the 1 to 8 lines of file2")

def modify_pcd_folders(folder_path, folder_transformed_path):
    for point_cloud_filename, point_cloud_transformed_filename in zip(
            sorted(os.listdir(folder_path)), sorted(os.listdir(folder_transformed_path))):

        print("files to be modified: ", point_cloud_filename ,", ", point_cloud_transformed_filename)
        modify_files(os.path.join(folder_path, point_cloud_filename), 
                    os.path.join(folder_transformed_path, point_cloud_transformed_filename))


# Call the function with two folder path that have similar substrings in their names
modify_pcd_folders("/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched/", 
                    "/home/osobky/Documents/Master/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/transformed/")




# Classical Point-to-point ICP for Vehicle-Infrastructure Localization


## Content
- The classical P2P ICP for vehicle-infrastrucutre localizaiton.
- Mulitple scripts for cleaning, changeing format & remove ground from data. for example: 
    - [convert_npy_to_pcd.py](open3d_registeration/convert_npy_to_pcd.py)
    - [pcd_to_bin.py](open3d_registeration/pcd_to_bin.py)
    - [remove_ground_save.py](open3d_registeration/remove_ground_save.py)

Check rest of scripts within open3d_registeration folder.

## Prerequistes 
- Data folders for Vehicle & Infrastructure. Refer to Data section in [Veh-Infra](https://) 

## Running the Algorithm
Follow the followng instructions:

1. Change directory to the P2P ICP
```bash
cd classical-p2p-icp/
```

2. Following the same pattern in the [cmd.txt](classical-p2p-icp/open3d_registeration/cmd) file. This contain different examples for different experiments. 

For example: 

```bash

python open3d_registeration/point2point_reg.py \
--src-dir /mnt/c/Users/elsobkyo/Documents/masters-thesis/Data/01_scene_01_omar/01_lidar/01_vehicle_lidar_robosense/vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched \
--trgt /mnt/c/Users/elsobkyo/Documents/masters-thesis/veh-infr-loc/local_map/kiss_icp_infra.pcd \
--offline \
--output-dir XX
```


# Point cloud registration

Point cloud registration with intensity

## Run



```bash


# play a rosbag
rosbag play --clock -l xxxx.bag


# run regsitration node
python registration.py --source_pc /s110/lidar/ouster/north/points --target_pc /s110/lidar/ouster/south/points --init_vsize 2 --con_vsize 2


# visualization:
rviz
```


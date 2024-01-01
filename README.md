# Vehicle-Infrastructure Localization

## Description
This repository hosts the implementation of two LiDAR odometry algorithms: Classical Point-to-Point ICP (Iterative Closest Point) and Modified KISS-ICP (Keep It Small and Simple ICP). It also includes comprehensive experiment scripts and data to evaluate and compare these algorithms in the context of vehicle-infrastructure localization.

## Folder Structure
- `/classical-p2p-icp`: Contains the implementation of the Classical Point-to-Point ICP algorithm. [Read more](./classical-p2p-icp/README.md).
- `/kiss-icp`: This submodule includes the modified KISS-ICP algorithm implementation. [Read more](./kiss-icp/README.md).
- `/data`: Data files used in the experiments. This directory may have further organization based on the nature of the data.
- `/experiments`: Consists of scripts and results of different experiments and evaluations. [Read more](./experiments/README.md).
- `/resources`: Contains resources like images, additional documentation.

## Getting Started

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/) (recommended)


### Cloning the Repository
To clone this repository and its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/OSobky/veh-infr-loc.git
```

### Dependencies by Conda
Create conda environment from `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash 
conda activate o3d
```

### Data 

For Vehicle-Infrastructure Localization we need to two sources of data:
- Vehilce point clouds
- Infrastrucutre point clouds

Plus: 
- GPS and IMU data from Vehilce points clouds
- Ground turth for transformation matrix between vehicle and infrastrucutre for evaluation purposes 

#### in this master thesis we used Scene one from a recorded data using roadside infrastructure in Munich. To access this data please contact [Walter Zimmer](walter.zimmer@tum.de)

#### Folder Structure 

As mentioned before for this thesis, we used scene one from recorded data. here is the folder structure for the recorded data: 

```
Data/
└── 01_scene_01_omar/
    ├── 01_lidar/
    │   ├── 01_vehicle_lidar_robosense/
    │   │   └── vehicle_lidar_robosense_driving_direction_east/s110_first_east/matched/
    │   └── 02_infrastructure_lidar_ouster/
    │       └── s110_lidar_ouster_south_driving_direction_east/matched/
    ├── 03_gps/
    │   └── 04_gps_position_drive/json
    ├── 04_imu/
    │   └── 04_imu_rotations_drive/json/matched
    └── tf_matrix/
        └── kitti/fixed_gt.txt
    
```

### Running the Algorithms
Follow the instructions in each algorithm's `README.md` for detailed setup and usage guidelines.

## Experiments
The `/experiments` folder contains scripts and datasets for evaluating the algorithms. For detailed information on running these experiments and understanding their results, refer to the [experiments README](./experiments/README.md).
<!-- 
## Contributing
We welcome contributions to this project. If you're interested in improving the algorithms or adding new features, please read our [contribution guidelines](./CONTRIBUTING.md). -->
<!-- 
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Contact
For any queries or further information, please contact [Omar Elsobky](mailto:omarelsobky97@gmail.com).



# Vehilce-Infrastrucutre Localization

## Description
This repository hosts the implementation of two LiDAR odometry algorithms: Classical Point-to-Point ICP (Iterative Closest Point) and Modified KISS-ICP (Keep It Small and Simple ICP). It also includes comprehensive experiment scripts and data to evaluate and compare these algorithms in the context of vehicle-infrastructure localization.

## Folder Structure
- `/classical-p2p-icp`: Contains the implementation of the Classical Point-to-Point ICP algorithm. [Read more](./classical-p2p-icp/README.md).
- `/kiss-icp`: This submodule includes the modified KISS-ICP algorithm implementation. [Read more](./kiss-icp/README.md).
- `/data`: Data files used in the experiments. This directory may have further organization based on the nature of the data.
- `/experiments`: Consists of scripts and results of different experiments and evaluations. [Read more](./experiments/README.md).
- `/resources`: Contains resources like images, additional documentation.

## Getting Started
To clone this repository and its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/OSobky/veh-infr-loc.git
```

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



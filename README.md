# SSC Exploration
This repository contains the code for **SC-Explorer**, our framework for incremental fusion of 3D semantic scene completion and safe and efficient integration thereof into mobile robotic mapping and planning for exploration.

![SC-Explorer](https://user-images.githubusercontent.com/36043993/191210581-530756ed-81f0-4413-8e21-9add00f39450.gif)

Incremental 3D scene completion for safe and efficient exploration mapping and planning.

# Table of Contents
**Credits**
* [Paper](#Paper)
* [Video](#Video)
   

**Setup**
* [Installation](#installation)
* [Simulation](#Simulation)

**Examples**
- [Training the Model](#training-the-model)
- [Running the Planner](#running-the-planner)
- [Evaluating an experiment](#evaluating-an-experiment)

# Paper
If you find this useful for your research, please consider citing our paper:

* Lukas Schmid, Mansoor Nasir Cheema, Victor Reijgwart, Roland Siegwart, Federico Tombari, and Cesar Cadena, "**SC-Explorer: Incremental 3D Scene Completion for Safe and Efficient Exploration Mapping and Planning**" in *ArXiv Preprint*, 2022.
  \[ [ArXiv](https://arxiv.org/abs/2208.08307) | [Video](https://youtu.be/DMXdhCqUqts)\]
  ```bibtex
  @article{schmid2022scexplorer,
    title={SC-Explorer: Incremental 3D Scene Completion for Safe and Efficient Exploration Mapping and Planning},
    author={Schmid, Lukas and Cheema, Mansoor Nasir and Reijgwart, Victor and Siegwart, Roland and Tombari, Federico and Cadena, Cesar},
    journal={arXiv preprint arXiv:2208.08307},
    year={2022}
  }
  ```

# Video
An overview of SC-Explorer is available on [YouTube](https://youtu.be/DMXdhCqUqts):

[<img src=https://github.com/ethz-asl/ssc_exploration/assets/36043993/09c4be47-4842-4bd0-8029-c41af455f7a8 alt="Youtube Video">](https://youtu.be/DMXdhCqUqts)



# Setup
> **ℹ️ Note**<br> The code is provided on an 'as-is' basis. While everything should work and is tested, photorealistic simulation and GPU inference can be cumbersome and we cannot provide support for setting this up.

## Installation
1. Install [ROS](http://wiki.ros.org/ROS/Installation) (Desktop full recommended) if not already done so.

2.  Install system dependencies: 
```shell script
sudo apt install python-wstool python-catkin-tools ros-$ROS_DISTRO-cmake-modules ros-$ROS_DISTRO-control-toolbox ros-$ROS_DISTRO-joy ros-$ROS_DISTRO-octomap-ros ros-$ROS_DISTRO-geographic-msgs autoconf libyaml-cpp-dev protobuf-compiler libgoogle-glog-dev liblapacke-dev libgeographic-dev
```

3. Setup catkin workspace using [catkin-tools](https://catkin-tools.readthedocs.io/en/latest/)
```shell script
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO 
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel
```

5. Checkout the GitHub repository: 
```shell script
cd ~/catkin_ws/src
git clone git@github.com:ethz-asl/ssc_exploration.git # Requires Git SSH
```

6. Install dependencies
```shell script
wstool init . ./ssc_exploration/.rosinstall # Requires Git SSH
wstool update
```
8. Compile
```shell script
catkin build ssc_planning
```

9. Build the scene completion network based on PAL-Net following the instructions in [`ssc_network/`](https://github.com/ethz-asl/ssc_exploration/tree/main/ssc_network).

## Simulation 
The following tools were used in our experiments, other simulation or real robot setups should work, too:

* **Unreal Engine**
Setup [Unreal Engine 4.25.6](https://www.unrealengine.com/en-US/download) (UE4).

* **Airsim**
[AirSim](https://microsoft.github.io/AirSim/) is a simulation software for simulating a MAV in Unreal Engine. We use the Airsim 1.2 plugin for UE4, which is provided by unreal_airsim below.

* **unreal_airsim** 
[unreal_airsim](https://github.com/ethz-asl/unreal_airsim) is a ROS  interface to the simulated MAV in Unreal Engine for accessing odometry and sending trajectory commands. 

* **Dataset**
The environment used in our experiments, including all plugins and ground truth can be downloaded [here](https://drive.google.com/drive/folders/1ji11IMJPlsnZQZmM4xNB3s9hnNgV0Ete?usp=sharing). Note that we can not guarantee compatibility with operating systems other than Ubuntu or different versions of AirSim/UE4.

# Experiments 

## Training the Model
Instructions to train and test the SC-Network as well as pretrained weights are given in [`ssc_network/`](https://github.com/ethz-asl/ssc_exploration/tree/main/ssc_network). 


## Running the Planner

1. Start Unreal Engine: 
```shell script
cd <UNREAL_INSTALL_DIR> # Move to the unreal install directory
./Engine/Binaries/Linux/UE4Editor your-project-file.uproject -opengl4
```

2. Start the SSC Network (Make sure the network is setup as explained in [`ssc_network/`](https://github.com/ethz-asl/ssc_exploration/tree/main/ssc_network) first):
```shell script
export SSC_DIR=/home/$USER/catkin_ws/src/ssc_exploration/ssc_network
python3 $SSC_DIR/infer_ros.py --model palnet --resume $SSC_DIR/pretrained_models/weights/PALNet.pth.tar
```

3. Launch planning pipeline with the desired planning configuration specified as argument:
 ```shell script
 roslaunch ssc_planning run.launch planner_config_file:=sc_explorer.yaml output_directory:=path/to/output
```
Feel free to play with parameters in `sc_explorer.yaml` or use different gain evaluators by specifying them in the config. Note that all configs are composed of two files, where `baseline` specifies the shared parameters and `sc_explorer` or `exploration` provide specialized parameters for view planning.

## Evaluating an experiment
Running the pipeline as explained above will write all output files into a timestamped folder in `path/to/output`. To evaluate the recorded data and visualize the map, run:

```shell script
python3 ./ssc_mapping/src/eval/eval_plots.py [output_dir] [gt_file_path] [eval_type]
```
Where `output_dir` should point to the previously created data directory, `gt_file_path` points to the ground truth (available [here](https://drive.google.com/drive/folders/1ji11IMJPlsnZQZmM4xNB3s9hnNgV0Ete?usp=sharing) for our simulated scene), and `eval_type` is one of `tsdf` or `hierarchical` to only evaluate measured or also scene completed areas, respectively.

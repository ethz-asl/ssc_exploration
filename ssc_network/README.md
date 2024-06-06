# Semantic Scene Completion

## Credits

The scene completion network is taken and adapted from PALNet [[Paper](https://ieeexplore.ieee.org/document/8902045) | [Original Code](https://github.com/waterljwant/SSC)]

    @article{li2019palnet,
	  title={Depth Based Semantic Scene Completion With Position Importance Aware Loss},
	  author={Li, Jie and Liu, Yu and Yuan, Xia and Zhao, Chunxia and Siegwart, Roland and Reid, Ian and Cadena, Cesar},
	  journal={IEEE Robotics and Automation Letters},
	  volume={5},
	  number={1},
	  pages={219--226},
	  year={2019},
	  publisher={IEEE}
    }


**A 3D Convolutional Neural Network for semantic scene completions from depth maps**
![3d_palnet_cnn](https://user-images.githubusercontent.com/10983181/148416145-ecc6f019-f7a2-47c1-9c30-6b0261cd4d89.png)

## Table of Contents
* [Installation](#installation)
* [Data Preparation](#Data-Preparation)
* [Train and Test](#Train-and-Test)
* [Inference (ROS)](#Inference)

## Installation
### Requirements:
- [pytorch](https://pytorch.org/)≥1.4.0
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
- imageio
- scipy
- scikit-learn
- tqdm

You can install the requirements by running `pip install -r requirements.txt`.

If you use other versions of PyTorch or CUDA, be sure to select the corresponding version of torch_scatter.


## Data Preparation
### Download dataset

The raw data can be found in [SSCNet](https://github.com/shurans/sscnet).

The repackaged data can be downloaded via 
[Google Drive](https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing)
or
[BaiduYun(Access code:lpmk)](https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg).

The repackaged data includes:
```python
rgb_tensor   = npz_file['rgb']		# pytorch tensor of color image
depth_tensor = npz_file['depth']	# pytorch tensor of depth 
tsdf_hr      = npz_file['tsdf_hr']  	# flipped TSDF, (240, 144, 240)
tsdf_lr      = npz_file['tsdf_lr']  	# flipped TSDF, ( 60,  36,  60)
target_hr    = npz_file['target_hr']	# ground truth, (240, 144, 240)
target_lr    = npz_file['target_lr']	# ground truth, ( 60,  36,  60)
position     = npz_file['position']	# 2D-3D projection mapping index
```

### 

## Train and Test

### Configure the data path in [config.py](https://github.com/ethz-asl/ssc_exploration/blob/main/ssc_network/config.py#L9)

```
'train': '/path/to/your/training/data'

'val': '/path/to/your/testing/data'
```

### Train
Edit the training script [run_SSC_train.sh](https://github.com/ethz-asl/ssc_exploration/blob/main/train.sh#L4), then run
```
bash train.sh
```

### Test
Edit the testing script [run_SSC_test.sh](https://github.com/ethz-asl/ssc_exploration/blob/main/test.sh#L3), then run
```
bash test.sh
```

## Inference
The SSC Network is deployed as ROS node for scene completions from depth topics. Please follow the follow instructuon for setting up ROS scene completion node.
### Pre-Requisites
* [**ROS**](http://wiki.ros.org/ROS/Installation)
* [**VoxelUtils**](https://github.com/ethz-asl/ssc_exploration/tree/main/voxel_utils)

   A python library providing C++ (CPU/CUDA) backend implementations for:
     - Fixed size TSDF volume computation from a single depth image
     - Fixed size 3D Volumetric grid computation by probabilistically fusing pointcloud (for SCFusion)
     - 3D projection indices from a 2D depth image 
> **_NOTE:_**  CUDA 10.2 is required for GPU backend.

Install the python extension from `voxel_utils` folder for inference on depth images from ROS topics:
```bash
cd voxel_utils
make #compile C++/CUDA code
pyhton setup.py install # install the package into current python environment
```

### Launching Scene Completion ROS node
```
python infer_ros.py --model palnet --resume trained_model.pth
```
A pretrained model can be download from [here](https://github.com/ethz-asl/ssc_exploration/blob/main/pretrained_models/weights/PALNet.pth.tar).

> **_NOTE:_** Make sure to source catkin workspace before starting inference. 

<div align="center">

# PointNet and PointNet++ for Classification and Part Segmentation on Point Clouds

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

I started this project with the aim to learn and understand the key ideas of the two great papers: PointNet and PointNet++. 

This repository is created on top of **[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)**, 
which comprises PyTorch Lightning (a lightweight PyTorch wrapper for high-performance AI research) and Hydra (a framework for elegantly configuring complex applications with hierarchical yaml files). 


## Datasets

#### ShapeNet Core Dataset

The ShapeNet Core dataset is a curated subset of the full ShapeNet dataset with single clean 3D models and manually
verified category and alignment annotations.

Download the [ShapeNet Core](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) dataset and save it to 'data/shapenetcore_normals' of the project root directory.

This subset contains 16 object categories. It is divided into 6068 training samples, 935 validation samples, and 1437 test samples.
Each point cloud sample includes coordinates with normals and per-pixel labels, enabling classification and part segmentation.


#### Modelnet40 Dataset

The ModelNet40 is a widely used dataset contains synthetic object point clouds for point cloud analysis. 
ModelNet40 has various categories, clean shapes, well-constructed dataset, etc. 
The original ModelNet40 consists of 12,311 CAD-generated meshes in 40 categories (such as airplane, car, plant, lamp), 
of which 9,843 are used for training while the rest 2,468 are reserved for testing. 
The corresponding point cloud data points are uniformly sampled from the mesh surfaces, and then further preprocessed by moving to the origin and scaling into a unit sphere.

Download the [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) dataset and save it to 'data/modelnet40_normal_resampled' of the project root directory.


## Networks

### PointNet

The main PointNet network is illustrated in the following figure. 




### PointNet++



## Installation

I recommend to use pip to create a virtual environment. I tried to. 
Both Anaconda and 
The environment can be created with Anaconda and `pip` virtual environment. 
Personally, I think `pip` seems to be 

#### Pip

```bash
# clone project
git clone https://github.com/willzhengwang/pointnet_plus_lightning.git
cd pointnet_plus_lightning

# [OPTIONAL] create a virtual environment with python=3.8 (or after. I used python 3.8.)

# Install a certain pytorch version according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

(Personal experience: `pip` seems it seems to be easier )

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## References

Github Repo: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

Github Repo: [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) <br>

Qi, C.R., Su, H., Mo, K. and Guibas, L.J., 2017. 
Pointnet: Deep learning on point sets for 3d classification and segmentation. 
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 652-660).

Qi CR, Yi L, Su H, Guibas LJ. 
Pointnet++: Deep hierarchical feature learning on point sets in a metric space. 
Advances in neural information processing systems. 2017;30.

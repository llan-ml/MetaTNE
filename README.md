# MetaTNE

This repository is the official implementation of MetaTNE in our paper [Node Classification on Graphs with Few-Shot Novel Labels via Meta Transformed Network Embedding](https://arxiv.org/abs/2007.02914).

## Requirements

We recommend first installing [Anaconda3-5.2.0](https://repo.anaconda.com/archive/). Then, run the following commands to install requirements:
```
pip install -U pip && pip uninstall -y numpy && pip install --ignore-installed wrapt numpy==1.17.3 tensorflow-gpu==2.0.0 && pip install networkx==2.2 ray[tune]==0.8.3
```

To better understand our code, please familiarize yourself with the usage of [Ray](https://github.com/ray-project/ray) and [Tune](https://github.com/ray-project/ray/tree/master/python/ray/tune).

## Usage

You can reproduce the results on BlogCatalog dataset as follows:

- Data Preparation
```
python standardize_data.py --data BlogCatalog
```

- Training and Evaluation
```
python run.py --dataset_str BlogCatalog --meta_num_pos_nodes 10 --meta_num_neg_nodes 20
```
You may need to modify [`num_gpus`](run.py#L33) and [`resources_per_trial`](run.py#L49) according to your computing resources.

- Result Analysis
```
python analysis.py --logdir ~/ray_results/BlogCatalog_10_20
```

# CLD: Unsupervised Feature Learning by Cross-Level Instance-Group Discrimination.

by Xudong Wang, Ziwei Liu and Stella X. Yu at UC Berkeley / ICSI and NTU. 

<em>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.</em>

<p align="center">
  <img src="http://people.eecs.berkeley.edu/~xdwang/projects/CLD/CLD.png"  width="60%" >
</p>

For more information, please check: [Project Page](http://people.eecs.berkeley.edu/~xdwang/projects/CLD/) | [PDF](http://people.eecs.berkeley.edu/~xdwang/papers/CVPR2021_CLD.pdf) | 
[Preprint](https://arxiv.org/abs/2008.03813v4) | [BibTex](http://people.eecs.berkeley.edu/~xdwang/papers/CLD.txt)

## Updates
[06/08/2021] Training and linear evaluating InfoMin + CLD on ImageNet is supported.

[05/20/2021] Training and linear evaluating MoCo v2 + CLD on ImageNet is supported.

[04/12/2021] Training MoCo + CLD on CIFAR is supported now.

[04/09/2021] Initial Commit. Training NPID + CLD on CIFAR is avaliable now in this repo. We also plan to support MoCo+CLD, BYOL+CLD and InfoMin+CLD.

## Requirements
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* pandas
* numpy
* [apex](https://github.com/NVIDIA/apex) (optional, unless using mixed precision training)

## Dataset Preparation
CIFAR and STL-10 code will download data automatically with the dataloader. For ImageNet, please download the ImageNet-1k dataset from [here](http://image-net.org/download). Moving validation images to labeled subfolders using the following script is required: [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). For ImageNet-100, it was firstly used in CMC and contains 100 categories of ImageNet. The category list of ImageNet-100 can be found in data/imagenet100.txt. Please organize ImageNet-100 according to the following structure. For high-correlation dataset Kitchen-HC, it is constructed by extracting objects in their bounding boxes from the multi-view RGB-D Kitchen dataset. Kitchen-HC can be downloaded from [this link](https://drive.google.com/drive/folders/1GDJ47C81kejmPU_tC7q2FkleGiGtYFmn?usp=sharing).
```
data
├── CIFAR-10
│   └── cifar-10-batches-py
│       ├── batches.meta
│       ├── data_batch_1
│       ├── ...
│       ├── data_batch_5
│       └── test_batch
├── CIFAR-100
│   └── cifar-100-python
│       ├── file.txt~
│       ├── meta
│       ├── train
│       └── test
├── Kitchen-HC
│   ├── train
│   │   ├── n02869837
│   │   ├── ...
│   │   └── n02090622
│   └── test
│       ├── n02869837
│       ├── ...
│       └── n02090622
├── ImageNet-100
│   ├── train
│   │   ├── n02869837
│   │   ├── ...
│   │   └── n02090622
│   └── val
│       ├── n02869837
│       ├── ...
│       └── n02090622
└── ImageNet
    ├── train
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    └── val
        ├── n01440764
        ├── ...
        └── n15075141
```

## Training and Evaluation Instructions
### CIFAR-10 and CIFAR-100
#### NPID + CLD
```
bash scripts/train_cifar10_npid_cld.sh or bash scripts/train_cifar100_npid_cld.sh
```
| Method          | Projection Head   | CIFAR-10 | CIFAR-100 | 
| --------------  | ----------------  | ---------------- | ---------------- 
| NPID                                | Linear | 80.8 | 51.6
| **NPID+CLD (reported)**             | Linear | 86.7 | 57.5
| **NPID+CLD (reproduced)**           | Linear | 86.8 | 58.8

The model is trained with mixed precision (fp16) by default, it is necessary to install apex if you want to apply mixed precision training. The reproduced result is the average kNN accuracies of 3 runs.

#### MoCo + CLD
```
bash scripts/train_cifar10_moco_cld.sh or bash scripts/train_cifar100_moco_cld.sh
```
| Method          | Projection Head   | CIFAR-10 | CIFAR-100 | 
| --------------  | ----------------  | ---------------- | ---------------- 
| MoCo                                | Linear | 82.1 | 53.1
| **MoCo+CLD (reported)**             | Linear | 87.5 | 58.1
| **MoCo+CLD (reproduced)**           | Linear | N/A | 59.1
| **MoCo+CLD (reproduced)**           | NormLinear | N/A | 59.7

The model is trained with the proposed NormLinear as the projection head by default. Other settings are the same as NPID+CLD.

### ImageNet
#### MoCo v2 + CLD
train
```
bash scripts/imagenet/train_imagenet_mocov2_cld.sh
```
linear evaluation
```
bash scripts/imagenet/test_imagenet_moco_cld.sh
```
| Method            | Projection Head   | #epochs | Top-1 (%) | Models
| --------------    | ----------------  | ---------------- | ---------------- | ---------------- 
| MoCov2                   | MLP     | 200 | 67.5 | - 
| **MoCov2+CLD**           | MLP     | 200 | 69.2 | [link](https://drive.google.com/file/d/1LAd0YMaRLZtqjDTukf1OKyFTrGUt9YgP/view?usp=sharing) 
| **MoCov2+CLD**           | NormMLP | 200 | 70.0 | [link](https://drive.google.com/file/d/1Jc2_rJiFZF1PzNB7UPyzhzpIv_NfUuls/view?usp=sharing)  

It is necesary to change the DATA_DIR, SAVE_DIR and PRETRAINED_MODEL. All models are pretrained for 200 epochs and evaluated with standard augmentation, linear decay scheduler and SGD optimizer.

#### InfoMin + CLD
train
```
bash scripts/imagenet/train_imagenet_infomin_cld.sh
```
linear evaluation
```
bash scripts/imagenet/test_imagenet_infomin_cld.sh
```
| Method            | Projection Head   | #epochs | Top-1 (%) | Models
| --------------    | ----------------  | ---------------- | ---------------- | ---------------- 
| InfoMin           | MLP     | 100 | 67.4 | - 
| **CLD**           | MLP     | 100 | 69.5 | [link](https://drive.google.com/file/d/1THFDbMdDlC81LJ8ZjNxTIanYZ-Dj238g/view?usp=sharing) 
| **CLD**           | NormMLP | 100 | 70.1 | link
| **CLD**           | MLP     | 200 | 70.6 | link
| **CLD**           | NormMLP | 200 | 71.5 | [link](https://drive.google.com/file/d/18hs7B4eQQK03p-dRhvJsVkw34Lm5Sg_x/view?usp=sharing)  

Please change the DATA_DIR and PRETRAINED_MODEL before launching experiments.

## How to get support from us?
If you have any general questions, feel free to email us at `xdwang at eecs.berkeley.edu`. If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 


## Citation
If you find our work inspiring or use our codebase in your research, please cite our work.
```
@article{wang2020unsupervised,
    title={Unsupervised Feature Learning by Cross-Level Instance-Group Discrimination},
    author={Wang, Xudong and Liu, Ziwei and Yu, Stella X},
    journal={arXiv preprint arXiv:2008.03813},
    year={2020}
}
```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
Part of this code is based on [NPID](https://github.com/zhirongw/lemniscate.pytorch), [MoCo](https://github.com/facebookresearch/moco), [CMC](https://github.com/HobbitLong/CMC), [infoMin](https://github.com/HobbitLong/PyContrast) and [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup).

# Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation (SIPE)

<img style="zoom:100%" alt="framework" src='fig/framework1.png'>

The implementation of [**Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation**](https://arxiv.org/abs/2203.02909), Qi Chen, Lingxiao Yang, Jianhuang Lai, and Xiaohua Xie, CVPR 2022.

## Abstract
Weakly Supervised Semantic Segmentation (WSSS) based on image-level labels has attracted much attention due to low annotation costs. Existing methods often rely on Class Activation Mapping (CAM) that measures the correlation between image pixels and classifier weight. However, the classifier focuses only on the discriminative regions while ignoring other useful information in each image, resulting in incomplete localization maps. To address this issue, we propose a Self-supervised Image-specific Prototype Exploration (SIPE) that consists of an Image-specific Propythontotype Exploration (IPE) and a General-Specific Consistency (GSC) loss. Specifically, IPE tailors prototypes for every image to capture complete regions, formed our Image-Specific CAM (IS-CAM), which is realized by two sequential steps. In addition, GSC is proposed to construct the consistency of general CAM and our specific IS-CAM, which further optimizes the feature representation and empowers a self-correction ability of prototype exploration. Extensive experiments are conducted on PASCAL VOC 2012 and MS COCO 2014 segmentation benchmark and results show our SIPE achieves new state-of-the-art performance using only image-level labels.

## Environment

- Python >= 3.6.6
- Pytorch >= 1.6.0
- Torchvision

## Usage

#### Step 1. Prepare Dataset

- PASCAL VOC 2012: [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- MS COCO 2014: [Image](https://cocodataset.org/#home) and [Label](https://drive.google.com/file/d/1Pm_OH8an5MzZh56QKTcdlXNI3RNmZB9d/view?usp=sharing).

#### Step 2. Train SIPE

```bash
# PASCAL VOC 2012
bash run_voc.sh

# MS COCO 2014
bash run_coco.sh
```

#### Step 3. Train Fully Supervised Segmentation Models

To train fully supervised segmentation models, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) and [seamv1](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc).

## Results

| Dataset         | Model       | mIoU (Train) | Weight                                                       | Training log                     |
| --------------- | ----------- | ------------ | ------------------------------------------------------------ | -------------------------------- |
| PASCAL VOC 2012 | CVPR submit | 58.65        | [Download](https://drive.google.com/file/d/1-_GXZq-1gxcbR7FdY1888tnxBAE39R-P/view?usp=sharing) | [Logfile](log/sipe_voc.log)      |
| PASCAL VOC 2012 | This repo   | 58.88        | [Download](https://drive.google.com/file/d/1YYYYXleRperCUrhcU4pT1eXybhlUQedW/view?usp=sharing) | [Logfile](log/sipe_voc_rep.log)  |
| MS COCO 2014    | CVPR submit | 32.09        | [Download](https://drive.google.com/file/d/1qWLvgjyd9eunyWJPyj02HcDQciiMKMu0/view?usp=sharing) | [Logfile](log/sipe_coco.log)     |
| MS COCO 2014    | This repo   | 32.63        | [Download](https://drive.google.com/file/d/103gU8AmTDXSnebh2q9xihOSxw4yoPGZb/view?usp=sharing) | [Logfile](log/sipe_coco_rep.log) |

## Citation
```
@InProceedings{Chen_2022_CVPR_SIPE,
    author = {Qi Chen, Lingxiao Yang, Jianhuang Lai, and Xiaohua Xie},
    title = {Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation},
    booktitle = {Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```

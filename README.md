# Hypernet Based IQA Assessment

This is the source code for the Digital Image Processing Homework 1. 

## Dependencies

Pretrained models from huggingface.io is engaged in the proposed model. The source code should be downloaded in the directory due to download failure on my side. Or else some modifications should be done when loading the pretrained model from huggingface.

Part of the packages required are as follows:

- sentence_transformers
- pytorch
- torchvision
- scipy
- ...

## Usages

### Training & Testing on IQA databases

Our model is based on the paper  "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network," in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). It now supports datasets AGIQA-3K and AIGCIQA2023. The root directory of the dataset should be changed to local directories in train_test_IQA.py to successfully load the data.



Training and testing our model on the AGIQA-3K dataset.

```
python train_test_IQA.py
```

Available options for training are provided in train_test_IQA.py. Only text alignment related codes are provided since evaluations on quality and authenticity could be done with just relatively slight modifications (and also the file is too big to be provided).

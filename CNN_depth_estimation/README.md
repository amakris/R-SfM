# CNN depth estimation framework

Pytorch implementation of the CNN depth estimation framework inspired by [this work](https://github.com/1adrianb/face-alignment).

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)

## Requirements

In order to train the CNN model you'll need:
- `PyTorch >= 1.10.0`
- Specify your custom [Dataset Class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files). The `__getitem__` function must return the input data of shape `(N,Ci​,H,W)` and the ground truth depth of shape `(N,1,H,W)`, where `N` is the batch size, `Cin` the input channels and `H`, `W` the spatial dimensions height and width respectivly. 


## Usage

In the `train_cnn.py` file change the desired hyper-parameters. To run the training procedure after making the above changes run:
    `python train_cnn.py`
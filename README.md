# MultiTask-CNN
Multi-Task CNN for Pose-Invariant Face Recognition.

Xi Yin and Xiaoming Liu.

## Introduction
This repository contains the code to repeat the experiments on MultiPIE and CASIA-Webface as described in the paper. 
The major changes we have made in caffe is to split and merge batches based on the ground truth or estimated pose information. 

We observe that for multi-task learning, it helps to learn joint features for all tasks. The final FC layer acts like feature selector to select the features for each specific task, thus it results in disentangled features, as illustrated in the following figure.

<p align="center">
<img src="https://github.com/xiyinmsu/MultiTask-CNN/blob/master/imgs/concept.png" alt="Multi-task Learning", width="1000px"> 
</p>

## Usage
After combine with caffe, please check the folder `examples/` to see the prototxt files. 
We save data into HDF5 format. 

## Citation
If you found this code useful, please consider to cite:
```
@article{yin2018multi,
  title={Multi-task convolutional neural network for pose-invariant face recognition},
  author={Yin, Xi and Liu, Xiaoming},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={2},
  pages={964--975},
  year={2018},
  publisher={IEEE}
}
```

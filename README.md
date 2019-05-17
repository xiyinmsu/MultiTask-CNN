# MultiTask-CNN
Multi-Task CNN for Pose-Invariant Face Recognition.

Xi Yin and Xiaoming Liu.

## Introduction
This repository contains the code to repeat the experiments on MultiPIE and CASIA-Webface as described in the paper. 
The major changes we have made in caffe is to split and merge batches based on the ground truth or estimated pose information. 

We observe that for multi-task learning, it helps to learn joint features for all tasks. The final FC layer acts like feature selector to select the features for each specific task, thus it results in disentangled features, as illustrated in the following figure.




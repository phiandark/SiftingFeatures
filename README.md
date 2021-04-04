# Code for the paper "Sifting out the features: pruning dense networks to obtain local image masks"

This repository hosts code to reproduce all results in the paper "Sifting out the features: pruning dense networks to obtain local image masks" by F. Pellegrini and G. Biroli. 

The main purpose of this code is performing Iterative Magnitude Pruning (IMP, see [this paper](https://arxiv.org/abs/1803.03635)) on a small network trained to classify images from ImageNet.
Please refer to the paper for further details.

The code is written in Python 3 based on Tensorflow (1.xx). The dataset is not provided in this repository, but can be found [here](http://www.image-net.org/download-images).

The main code to run an IMP is contained in the `src` folder.

The code `IMP.py` can be run by supplying a `.ini` input file.
The structure and all keywords in the input file are described in `input.md`.
Sample input files for the main experiments presented in the paper can be found in the `inputs` folder.

# Code for the paper "Sifting out the features by pruning: Are convolutional networks the winning lottery ticket of fully connected ones?"

This repository hosts code to reproduce all results in the paper by F. Pellegrini and G. Biroli, that can be found here.

The main purpose of this code is performing Iterative Magnitude Pruning (IMP, see [this paper](https://arxiv.org/abs/1803.03635)) on a small network trained to classify images from ImageNet.
Please refer to the paper for further details.

The code is written in Python 3 based on Tensorflow (1.xx). The dataset is not provided in this repository, but can be found [here](http://www.image-net.org/download-images).

### Other content

This repository also contains other material for the paper in the folder [SupplementaryMaterial](SupplementaryMaterial):

- [ImageNet_10cl.txt](SupplementaryMaterial/ImageNet_10cl.txt) contains the "Meaningful" 10 super-classes used in the paper. It consists of 4 columns reporting for all the 1000 Imagenet categories: original name, original index, our index (in 0-9), and the short class description.
- [Masks_evolution.mp4](SupplementaryMaterial/Masks_evolution.mp4) is a movie with the evolution of the 225 most connected first layer masks (at the end of IMP) for the main experiment in the paper.
- [Masked_weights_evolution.mp4](SupplementaryMaterial/Masked_weights_evolution.mp4) is a movie of the same masks, but multiplied by the corresponding weight.



### Code details

The code to run an IMP is contained in the [src](src) folder.

The main code is called [IMP.py](src/IMP.py) and it can be run by supplying a `.ini` input file:
```
python IMP.py in.ini
```
The structure of the input file and all possible keywords are described in [input.md](input.md).
Sample input files for the main experiments presented in the paper can be found in the [inputs](inputs) folder.

The main purpose of the code is performing IMP starting from a FCNN training on ImageNet32 images.
The code is not particularly optimized and it does not exploit the progressively sparser structure of the networks.
However, with standard parameters the whole IMP procedure should run in a few hours on a modern GPU (e.g. RTX2080).

The code produces several output files:

- `IMP_out.dat` is a simple text file written during training reporting train and validation error (useful to monitor the training process)
- `IMP_itdata_x.pkl` contains summary data on iteration `x` of the IMP, to be loaded to analyze the network.
- `IMP_findata.pkl` is written at the end of the computation, with summary data of the whole IMP process.

A jupyter notebook [IMP_PostProcess.ipynb](src/IMP_PostProcess.ipynb) is provided with the correct structure to load the `.pkl` files, analyze the networks and produce all the plots shown in the main article.


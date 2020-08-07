# ElisaNet
## Deep Convolutional Neural Network for Elisa Image Classification

This is a simple implementation of a convolutional neural network for classifying Elisa images.

### Prerequisits
* Python > 3.0
* Pytorch >= 1.4
* Skimage >= 0.17.2
* Numpy >= 1.19.1

This repo includes a ```env.yml``` file for creating a conda environment that will work with this repo.

Model weights can be downloaded here: [link](https://drive.google.com/file/d/1wp-RO5Y-wBj8u-6TJJ8ia7669FEqL0Es/view?usp=sharing)

Place the ```.pth``` file in the same directory as the ```test.py``` file.

The network currently only accepts image resolution of 3648x2736 RGB format. Will be updated in the future to be more flexible. A sample of how the input images should is shown below.

<img src="sample.jpg" alt="drawing" width="342"/>

### Usage
To test a single image or multiple images, type the following in the terminal where ```test.py``` is located:
```
python test.py --images img1,img2,img3...
```
Remember to separate each image file with a comma WITHOUT spaces.

If using a GPU to run the test, add the ```--gpu index``` flag where ```index``` is the index of the gpu to run on.

For more help, type ```python test.py --help```.

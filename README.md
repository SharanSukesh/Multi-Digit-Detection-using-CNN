# Multi-Digit-Detection-using-CNN
<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Libraries used](#libraries-used)
* [Dataset used](#dataset-used)
* [Built on](#built-on)
* [Questions answered](#questions-answered)
* [Hypotheses Tested](#hypotheses-tested)
* [Ackowledgements](#ackowledgements)
* [Author](#author)


## About the Project 

In this notebook we will be using CNN on the MNSIT dataset in order to detect numbers with 1 or more digits from handwritten numbers. The model will be able to finally return the number that was shown to it. To do this, we will be using OpenCV in order to detect and highlight the digits that are identified. To do this we will use various different libraries such as keras, matplotlib and OpenCV. Lastly, we will save the model and create a simple app using Flask.

The python notebook __"Multi-Digit Detection Using CNN"__ contains an all the steps to import the dataset, format the images using image augmentation and finally train our model and make predictions on the images. 

## Libraries used 
* Numpy
* Pandas
* Matplotlib
* keras
* OpenCV

```bash
import cv2
import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from keras.datasets import mnist
```

## Dataset used 
* __Courant Institute, NYU, Google Labs, New York, Microsoft Research, Redmond__ - MNIST Dataset

## Built with
* Jupyter Notebook

## Model Training and Testing Steps
1. Preparing Training and Test Sets
2. Creating and Training the Model
3. Image Transformations
4. Predictions

## Ackowledgements
* <a href='http://yann.lecun.com/exdb/mnist/'>MNIST</a> - Dataset

## Author - Sharan Sukesh

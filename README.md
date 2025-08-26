Author: [Robert Schulz, University of Technology Chemnitz, Germany](mailto:schulzr256@gmail.com)

# MSCONV3D
A multistage 3D convolution-based model for Human Action Recognition

## Overview

In this work, we introduce MSCONV3D, a multistage 3D convolution-based model for HAR.

![](image/msconv3d.png)
_**Figure 1.** Illustration of the MSCONV3D model architecture. The model consists of multiple convolutional blocks (ConvBlock), each incorporating 3D convolution, pooling, and normalization layers. The output of each ConvBlock is down sampled through pooling and subsequently fed into the following layers, where it is concatenated with outputs from subsequent layers. This design enables multi-stage feature extraction, comparable to skip connections, allowing the model to retain spatial and temporal information across different scales. The final classification layers include a flattening operation, a fully connected layer, and a softmax activation to produce output probabilities._

# Training
![](validate/msconv3ds_rgb/confusion_matrix_rel.png)

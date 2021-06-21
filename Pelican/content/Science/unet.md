Title: Notes on U-Net Paper
Date: 2021-06-21 10:40
Tags: Neural Networks, AI, ML
Author: Cody Fernandez
Summary: Collecting the notes I took while reading the U-Net paper

# U-NET
- The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
    - localization: a class label is supposed to be assigned to each pixel
- Built upon paper "Fully convolutional networks for semantic segmentation" Long et. al [9]
- In [9], you supplement a usual contracting network by successive layers, replacing pooling operators with upsampling operators. These layers increase output resolution. To localize, high resolution features from contracting path are combined with teh upsampled output. a successive convolution layer then learns to assemble more precise output from this info
- Modification: in the upsampling part, we have a large number of feature channels. These allow the network to propagate context information to higher resolution layers. Now the expansive path is pretty much symmetric with teh contracting path (u-shape)
- U-net has no fully connected layers and only uses the "valid" part of each convolution. 
    - valid: segmentation map only contains pixels for which the full context is available in the input image.
- Seamless segmentiaon of arbitrarily large images through an overlap-tile strategy
- to predict border region pixels, mirror the input image to extrapolate the missing context.
    - reduces GPU memory issues
- Apply elastic deformations to the available training images
    - if your dataset is weak
    - value of data augementation for learning invariance [2]
- Word description of architecture
    - Contracting path (left) and expansive path (right)
        - contracting path is standard CNN. Repeat two 3x3 (unpadded) convolutions, followed by a ReLU and a 2x2 max pooling with stride 2 (for downsampling).
        - at each downsamplingk double the number of feature channels
    - expansive path, upsample the feature map, 2x2 "upconvolution" to halve the number of feature channels, concatenate with cropped feature map from contracting path, two 3x3 convolutions, two ReLUs
    - final layer, 1x1 convolution to map each 64-component feature vector to teh number of classes. 23 convolution layers in total
- For seamless tiling of output segmentation map, select input tile size such that all 2x2 max-poolings are applied to a layer with even x- and y-size.
- The output image is smaller that the input by a constant border width due to the unpadded convolutions.
- Favor large input tiles over large batch size
    - batch size of 1 single image
- high momentum (.99) so a large number of previously seen training samples determine the update in the current optimization step
- Energy function: pixel-wise soft-max over the final feature map combined with the cross entropy loss function
- soft-max: $p_k(\vec{x})=\frac{e^{a_k(\vec{x})}}{\sum_{k'=1}^{k}e^{a_{k'}(\vec{x})}}$
    - $a_k(\vec{x}$: activation in feature channel $k$ at pixel position $\vec{x} \in \Omega$ with $\Omega \subset \mathbb{Z}^2$
    - $k$: number of classes
    - $p_k(\vec{x})$: approximated maximum function
        - $p_k(\vec{x}) \approx 1$ for the $k$ that has the maximum activation $a_k(\vec{x})$
        - $p_k(\vec{x}) \approx 0$ for all other $k$
- cross entropy then penalizes at each position the deviation of $p_l(\vec{x})$ from 1 by $E=\sum_{\vec{x} \in \Omega} W(\vec{x}) \log (p_l(\vec{x})(\vec{x}))$
    - $l$: $\Omega \rightarrow {1,...,k}$ the true label of each pixel
    - $w$: $\Omega \rightarrow \mathbb{R}$ a weight map we introduce to givbe some pixels more importance during training
- pre-compute the weight map for each ground truth segmentation to compensate the different frequency of pixels from a certain class in the training set
    - also to force the network to learn teh small separation borders introduced between touching cells (always necessary?)
- Compute separation border using morphological operations 
- weight map: $w(\vec{x})=w_c(\vec{x})+w_0 \exp(-\frac{(d_1(\vec{x})+d_2(\vec{x}))^2}{2\sigma_2})$
    - $w_c$: $\Omega \rightarrow \mathbb{R}$ weight map to balance the class frequencies
    - $d_1$: $\Omega \rightarrow \mathbb{R}$ distance to the border of the nearest cell
    - $d_2$: $\Omega \rightarrow \mathbb{R}$ distance to the border of the second nearest cell
- set $w_0=10$ and $\sigma \approx 5$ pixels
- Initialize weights so feature map has unit variance. Use a Gaussian with standard deviation of $\sqrt{\frac{2}{N}}$, $N$ the number of incoming nodes of one neuron.
### This is a tutorial on Automatic Segmentation of the claustrum in MR images using deep learning. 

We provide pre-trained models to directly perform inference for your new dataset. Here is a [simple demo](https://drive.google.com/file/d/1b0XS8LjRM-rZMPOL8qM6voG-A5jcdUgK/view?usp=sharing) to test on new dataset.

The process mainly includes four steps:
a) the pre-processing of the images.  <br />
b) training two single-view convolutional neural networks (U-Net) to segment the claustrum.  <br />
c) ensemble of the results from single views.  <br />
d) simple post-proccesing of the results. 

### This is a tutorial on Automatic Segmentation of the claustrum in MR images using deep learning. 

We provide pre-trained models to directly perform inference for your new dataset. Here is a [simple demo](https://drive.google.com/file/d/1b0XS8LjRM-rZMPOL8qM6voG-A5jcdUgK/view?usp=sharing) to test on new dataset.

There are several pre-processing steps you will need to follow before you feed your data to our pre-trained model:
a) resampling the MR scans to 1 mm^3;
b) skull-stripping;
c) image denoising using an adaptive nonlocal means filter for 3D MRI ([ANLM](https://sites.google.com/site/pierrickcoupe/softwares/denoising-for-medical-imaging/mri-denoising)). Default setting was used in our work.  <br />

After these steps, you could play around with the test codes. 


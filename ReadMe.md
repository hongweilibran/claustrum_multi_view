
## A tutorial on automatic segmentation of the claustrum in MR images using deep learning. 

### Test your dataset

<br />

We provide pre-trained models to directly perform inference for your new dataset. Here is a [simple demo](https://drive.google.com/file/d/1b0XS8LjRM-rZMPOL8qM6voG-A5jcdUgK/view?usp=sharing) to test on new dataset. The codes can run without GPUs.  The detailed instructions are in **ReadMe** inside. Please have a look at it. <br /> 
<br />
There are several pre-processing steps you will need to follow before you feed your data to our pre-trained model: <br /> <br />
a) Resampling the MR scans to 1 mm^3; We provide the python codes for resampling in the repository. <br />
b) Skull-stripping; There are two options: i) [ROBEX](https://www.nitrc.org/projects/robex), and ii) FSL-BET. All works well. We provide simple bash files to use FSL-BET <br />
c) Image denoising using an adaptive nonlocal means filter for 3D MRI ([ANLM](https://sites.google.com/site/pierrickcoupe/softwares/denoising-for-medical-imaging/mri-denoising)). Default setting was used in our work.  <br /> <br />

After these steps, you could play around with the test codes. 

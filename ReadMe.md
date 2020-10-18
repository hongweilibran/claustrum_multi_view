
## A tutorial on automatic segmentation of the claustrum in MR images using deep learning. 

### How to directly apply it on your own dataset with our pre-trained models?

<br />

We provide **pre-trained** models trained on 200 T1-w scans to directly perform inference for your new datasets.  <br /> 
Here is a [simple demo](https://drive.google.com/file/d/1b0XS8LjRM-rZMPOL8qM6voG-A5jcdUgK/view?usp=sharing) to test on new dataset. We use a public dataset from [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/). Basically the codes take 3D T1-w scans as the input and give you a 3D segmentation mask. The codes can run without GPUs.  The detailed instructions are in **ReadMe** inside. Please have a look at it. <br /> 
<br />
There are several **necessary** pre-processing steps you will need to follow before you feed your data to our pre-trained model: <br /> <br />
a) **Resampling** the MR scans to 1 mm^3; This is because we trained the model on this kind of resolution. We provide the python codes for resampling in the repository. <br />
b) **Skull-stripping**; We tried two options: i) [ROBEX](https://www.nitrc.org/projects/robex), or ii) FSL-BET. All works well. We provide a simple bash file to use FSL-BET to deal with a couple of images in a loop. <br />
c) **Image denoising** using an adaptive nonlocal means filter for 3D MRI ([ANLM, in Matlab](https://sites.google.com/site/pierrickcoupe/softwares/denoising-for-medical-imaging/mri-denoising)). Unfortunately we did not find the python version for this step. The default setting in Matlab was used in our work.  <br /> <br />

After these three steps, you could play around with the test codes. Feel free to ask [me](http://campar.in.tum.de/Main/HongweiLi) any questions.  <br />


### How to train your own models

a) Data preparation. Resampling is not necessary if you want to work on the resolution you prefer. <br />
b) Training.


### Disclaimer on Datasets

We do not host or distribute [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

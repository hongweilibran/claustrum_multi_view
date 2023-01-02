# test a model
# predict a segmentation and save it in 'output/' as '..._pred_mask.nii.gz'
# you can set the variable "combined_model" to True or False depending on your intention: 
# testing a combination of three axial and three coronal models or testing one single model (axial or coronal)


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import SimpleITK as sitk
import scipy
import scipy.spatial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from PIL import Image
from pathlib import Path

from train_model import pre_processing, inverse_orient, check_orient, dice_coef_for_training, dice_coef_loss, conv_bn_relu, get_crop_shape, get_unet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K.set_image_data_format('channels_last')
smooth=1.

per = 0.25
thresh = 10
img_shape = (200, 200, 1)

## crop or pad the volumes to be a standard size given the original size
def crop_or_pad(input_array, per_, ori_size):
   # dim_1 = np.shape(input_array)[0]
    dim_2 = np.shape(input_array)[1]
    dim_3 = np.shape(input_array)[2]
    rows = ori_size[1]
    cols = ori_size[2]
    array_1 = np.zeros(ori_size, dtype = 'float32')
    array_1[...] = np.min(input_array)
    
    if dim_2 <=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows - dim_2)/2):(int((rows - dim_2)/2)+ dim_2), int((cols - dim_3)/2):(int((cols - dim_3)/2)+dim_3)] = input_array[:, :, :, 0]
    elif dim_2>=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, :] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), int((dim_3-cols)/2):(int((dim_3-cols)/2)+cols), 0]
    elif dim_2>=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, int((cols-dim_3)/2):(int((cols-dim_3)/2)+dim_3)] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), :, 0]
    elif dim_2<=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows-dim_2)/2):(int((rows-dim_2)/2)+ dim_2), :] = input_array[:, :, int((dim_3 -cols)/2):(int((dim_3 -cols)/2)+cols), 0]
    return array_1


## simple post-processing
def post_processing(input_array, ori_size, orient_1):
   # output_array = np.zeros(ori_size, dtype= 'float32')
    output_array = crop_or_pad(input_array, per, ori_size)
    inv_orient = inverse_orient(orient_1)
    output_array = np.transpose(output_array, inv_orient)
    return output_array


## prediction of the label by one model
def simple_prediction(coronal_array, ori_size_c, orient_c, axial_array, ori_size_a, orient_a):
    model_path_0 = os.path.join(model_path, orientation+'_model'+str(model_num)+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_0)
    if orientation == 'c':
        pred_0 = model.predict(coronal_array, batch_size=1, verbose=True)
        pred_0_post = post_processing(pred_0, ori_size_c, orient_c)
    elif orientation == 'a':
        pred_0 = model.predict(axial_array, batch_size=1, verbose=True)
        pred_0_post = post_processing(pred_0, ori_size_a, orient_a)
    
    return pred_0_post


## prediction of the label by three coronal and three axial models
def combined_prediction(coronal_array, ori_size_c, orient_c, axial_array, ori_size_a, orient_a):
    print('combined prediction')
    #do inference on different views     
    model_path_1 = os.path.join(model_path,'c_model'+str(coronal_model_list[0])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_1)  
    pred_1c = model.predict(coronal_array, batch_size=1, verbose=True)
    model_path_2 = os.path.join(model_path,'c_model'+str(coronal_model_list[1])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_2)  
    pred_2c = model.predict(coronal_array, batch_size=1, verbose=True)
    model_path_3 = os.path.join(model_path,'c_model'+str(coronal_model_list[2])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_3)  
    pred_3c = model.predict(coronal_array, batch_size=1, verbose=True)
    # ensemble coronal predictions
    pred_c = (pred_1c+pred_2c+pred_3c)/3

    model_path_1 = os.path.join(model_path,'a_model'+str(axial_model_list[0])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_1)  
    pred_1a = model.predict(axial_array, batch_size=1, verbose=True)
    model_path_2 = os.path.join(model_path,'a_model'+str(axial_model_list[1])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_2)  
    pred_2a = model.predict(axial_array, batch_size=1, verbose=True)
    model_path_3 = os.path.join(model_path,'a_model'+str(axial_model_list[2])+'_epoch'+str(epoch_count)+'.h5')
    model.load_weights(model_path_3)  
    pred_3a = model.predict(axial_array, batch_size=1, verbose=True)
    # ensemble axial predictions
    pred_a = (pred_1a+pred_2a+pred_3a)/3
    
    # transform them to their original size and orientations
    pred_1_post = post_processing(pred_c, ori_size_c, orient_c)
    pred_2_post = post_processing(pred_a, ori_size_a, orient_a)

    # ensemble of two views
    pred = (pred_1_post+pred_2_post)/2

    return pred


def predict_label(pat, pat_count):
    #read data
    pat_file_name = os.path.join(data_path, pat)
    ref_image = sitk.ReadImage(pat_file_name)
    image_array = sitk.GetArrayFromImage(ref_image)
    
    # z-score normalization
    brain_mask_T1 = np.zeros(np.shape(image_array), dtype = 'float32')
    brain_mask_T1[image_array >=thresh] = 1
    brain_mask_T1[image_array < thresh] = 0
    for iii in range(np.shape(image_array)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside br
    image_array = image_array - np.mean(image_array[brain_mask_T1 == 1])
    image_array /= np.std(image_array[brain_mask_T1 == 1])
    
    # transform/project the original array to axial and coronal views
    coronal_array = np.transpose(image_array, (1, 0, 2))
    axial_array = image_array

    ## this is to check if the orientation of your images is right or not. Please check /images/coronal and /images/axial
    check_orient (coronal_array, direction_1, pat_count)
    check_orient (axial_array, direction_2, pat_count)
    
    #original size and orientations 
    ori_size_c = np.asarray(np.shape(coronal_array))
    ori_size_a = np.asarray(np.shape(axial_array))
    orient_c = [1, 0, 2]
    orient_a = [0, 1, 2]
    
    #pre-processing, crop or pad them to a standard size [N, 200, 200]
    coronal_array = pre_processing(coronal_array, per, img_shape)
    axial_array = pre_processing(axial_array, per, img_shape)
    
    if combined_model == False:
        pred = simple_prediction(coronal_array, ori_size_c, orient_c, axial_array, ori_size_a, orient_a)
    elif combined_model == True:
        pred = combined_prediction(coronal_array, ori_size_c, orient_c, axial_array, ori_size_a, orient_a)

    # threshold for empirical superior prediction
    pred[pred > 0.40] = 1.
    pred[pred <= 0.40] = 0.
    
    #save the masks
    filename_resultImage = os.path.join(result_path, pat)
    filename_resultImage = filename_resultImage.replace('.nii.gz','_pred_mask.nii.gz') # adapt this line if you don't use '.nii.gz' as file format
    sitk_image = sitk.GetImageFromArray(pred)
    sitk_image.CopyInformation(ref_image)
    sitk.WriteImage(sitk_image, filename_resultImage)
    
direction_1 = 'coronal'
direction_2 = 'axial'

result_path = 'output/'
image_path = 'images/'
data_path = 'data/test/'
pat_list = sorted(os.listdir(data_path))



## True for testing a combined model of three axial and three coronal models or False for testing a single model
combined_model = True

if combined_model == True:
    # set model_path = 'saved_models/' if you want to test models which were saved in saved_models/
    model_path = 'models/'
    # adapt the elements of the lists corresponding to coronal and axial model_num for your own models
    coronal_model_list = [0,1,2]  
    axial_model_list = [0,1,2]
    epoch_count = 30

## if you want to test a single model:
elif combined_model == False:
    # set model_path = 'saved_models/' if you want to test your own model which is saved in saved_models/
    # in this case you have to further adapt the following hyperparameters depending on your model name
    model_path = 'models/'
    orientation = 'a'        # 'a' for axial or 'c' for coronal models
    model_num = 0           # number of the model (e.g. model3 -> model_num=3)
    epoch_count = 30         # the epoch count is also part of the model name (e.g. epoch_count=30)

else:
    print('Please set combined_model to True or False')
    exit()


model = get_unet(img_shape)

def main():
    pat_count = 0
    for pat in pat_list:
        print(pat_count)
        print(pat)   
        predict_label(pat, pat_count)
        pat_count += 1


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:03:13 2019

@author: bran
"""



import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import time
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import SimpleITK as sitk
from evaluation import hd95, hd
K.set_image_data_format('channels_last')
import warnings
K.set_image_data_format('channels_last')
smooth=1.
import scipy

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

def get_unet(img_shape = None, first5=False):
        inputs = Input(shape = img_shape)
        concat_axis = -1

        if first5: filters = 5
        else: filters = 3
        conv1 = conv_bn_relu(32, filters, inputs)
        conv1 = conv_bn_relu(32, filters, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv_bn_relu(64, 3, pool1)
        conv2 = conv_bn_relu(64, 3, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = conv_bn_relu(96, 3, pool2)
        conv3 = conv_bn_relu(96, 3, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_bn_relu(128, 3, pool3)
        conv4 = conv_bn_relu(128, 4, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = conv_bn_relu(256, 3, pool4)
        conv5 = conv_bn_relu(256, 3, conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = conv_bn_relu(128, 3, up6)
        conv6 = conv_bn_relu(128, 3, conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = conv_bn_relu(96, 3, up7)
        conv7 = conv_bn_relu(96, 3, conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = conv_bn_relu(64, 3, up8)
        conv8 = conv_bn_relu(64, 3, conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = conv_bn_relu(32, 3, up9)
        conv9 = conv_bn_relu(32, 3, conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

        return model

def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = testImage.flatten()
    resultArray = resultImage.flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 

def getVS(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    v_1 = np.sum(testImage)
    v_2 = np.sum(resultImage)
    return (1- (float(abs(v_1 - v_2)) / float(v_1 + v_2) ))* 100
    

def crop_or_pad(input_array, per, ori_size):
   # dim_1 = np.shape(input_array)[0]
    dim_2 = np.shape(input_array)[1]
    dim_3 = np.shape(input_array)[2]
    rows = ori_size[1]
    cols = ori_size[2]
    array_1 = np.zeros(ori_size, dtype = 'float32')
    array_1[...] = np.min(input_array)
    
    if dim_2 <=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per): (ori_size[0] -int(ori_size[0]*per)), int((rows - dim_2)/2):(int((rows - dim_2)/2)+ dim_2), int((cols - dim_3)/2):(int((cols - dim_3)/2)+dim_3)] = input_array[:, :, :, 0]
    elif dim_2>=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per): (ori_size[0] -int(ori_size[0]*per)), :, :] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), int((dim_3-cols)/2):(int((dim_3-cols)/2)+cols), 0]
    elif dim_2>=rows and dim_3<=cols: 
        array_1[int(ori_size[0]*per): (ori_size[0] -int(ori_size[0]*per)), :, int((cols-dim_3)/2):(int((cols-dim_3)/2)+dim_3)] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), :, 0]
    elif dim_2<=rows and dim_3>=cols: 
        array_1[int(ori_size[0]*per): (ori_size[0] -int(ori_size[0]*per)), int((rows-dim_2)/2):(int((rows-dim_2)/2)+ dim_2), :] = input_array[:, :, int((dim_3 -cols)/2):(int((dim_3 -cols)/2)+cols), 0]
    return array_1

def inverse_orient(orient_):
    inv_orient = []
    if np.ndarray.tolist(orient_) == [0, 1, 2]:
        inv_orient = (0, 1, 2)
    elif np.ndarray.tolist(orient_) == [1, 0, 2]:
        inv_orient = (1, 0, 2)
    elif np.ndarray.tolist(orient_) == [1, 2, 0]:
        inv_orient = (2, 0, 1)
    elif np.ndarray.tolist(orient_) == [2, 1, 0]:
        inv_orient = (2, 1, 0)
    return inv_orient
        

def post_processing(input_array, ori_size, orient_1):
   # output_array = np.zeros(ori_size, dtype= 'float32')
    output_array = crop_or_pad(input_array, per, ori_size)
    inv_orient = inverse_orient(orient_1)
    output_array = np.transpose(output_array, inv_orient)
    return output_array



direction_1 = 'coronal'
direction_2 = 'axial'

per = 0.2

img_shape = (180, 180, 1)
model = get_unet(img_shape)
model_path = 'models/No_Augmentation/'
result_path = 'CV_all/'

CV_list_all = [[3, 3, 3, 3, 3], [9, 9, 9, 9, 10], [20, 20, 21, 21, 21], [3, 3, 3, 4, 4]]
if not os.path.exists(result_path):
    os.mkdir(result_path)
result_vs = []
result_dsc = []
result_h95 = []
for ss in range(4, 5):
    
    nameList = []
    for fold_num in range(0, 1):
        test_dir = 'segmentations/scanner_'+str(ss)+'/'
        CV_list = CV_list_all[ss-1]
        images_1 = np.load('data_'+str(ss)+'_'+direction_1+'.npy')
        masks_1 = np.load('gt_mask_'+str(ss)+'_'+direction_1+'.npy')
        images_2 = np.load('data_'+str(ss)+'_'+direction_2+'.npy')
        masks_2 = np.load('gt_mask_'+str(ss)+'_'+direction_2+'.npy')
        slices_id_1 = np.load('patient_id_'+str(ss)+'_'+direction_1+'.npy')
        slices_id_2 = np.load('patient_id_'+str(ss)+'_'+direction_2+'.npy')
        CV_indices = np.load('scanner_'+str(ss)+'_indices.npy')
        name_list = np.load(str(ss)+'_nameList'+'_new.npy')
        size_list_1 = np.load(str(ss)+'_o'+'_size_list.npy')
   #     size_list_2 = np.load(str(ss)+'_'+direction_2+'_size_list.npy')
        
        orientation_1 = np.load(direction_1+'_orient_'+str(ss)+'.npy')
        orientation_2 = np.load(direction_2+'_orient_'+str(ss)+'.npy')
        
        if fold_num == 0:
            CV_indices_fold = CV_indices[0:CV_list[0]]
        else:
            CV_indices_fold = CV_indices[np.sum(CV_list[0:fold_num]): np.sum(CV_list[0:(fold_num+1)])]
        
        test_num = CV_list[fold_num]
        for ii in range(test_num):
            ori_size_c = np.zeros((3,1), dtype = 'int32')
            ori_size_a = np.zeros((3,1), dtype = 'int32')
            test_p_id = CV_indices_fold[ii]
            ori_size = size_list_1[test_p_id, :]
     #       ori_size_a = size_list_2[test_p_id, :]
            orient_c = orientation_1[test_p_id]
            orient_a = orientation_2[test_p_id]
            
            ori_size_c[0] = ori_size[orient_c[0]]
            ori_size_c[1] = ori_size[orient_c[1]]
            ori_size_c[2] = ori_size[orient_c[2]]
            ori_size_c = ori_size_c[:,0]
            
            ori_size_a[0] = ori_size[orient_a[0]]
            ori_size_a[1] = ori_size[orient_a[1]]
            ori_size_a[2] = ori_size[orient_a[2]]
            ori_size_a = ori_size_a[:,0]
            
            test_images_1 = images_1[slices_id_1[:,0] == test_p_id, ...]
            test_gt_1 = masks_1[slices_id_1[:,0] == test_p_id, ...]
            test_images_2 = images_2[slices_id_2[:,0] == test_p_id, ...]
            test_gt_2 = masks_2[slices_id_2[:,0] == test_p_id, ...]
            
            model_path_1 = model_path+ 'CV_all_'+direction_1+'_'+ str(fold_num)+ '_0.h5'
            model.load_weights(model_path_1)  
            pred_1c = model.predict(test_images_1, batch_size=1, verbose=True)
            model_path_1 = model_path+ 'CV_all_'+direction_1+'_'+ str(fold_num)+ '_1.h5'
            model.load_weights(model_path_1)  
            pred_2c = model.predict(test_images_1, batch_size=1, verbose=True)
            model_path_1 = model_path+ 'CV_all_'+direction_1+'_'+ str(fold_num)+ '_2.h5'
            model.load_weights(model_path_1)  
            pred_3c = model.predict(test_images_1, batch_size=1, verbose=True)
            
            pred_c = (pred_1c+pred_2c+pred_3c)/3
            
            model_path_2 = model_path+ 'CV_all_'+direction_2+'_'+ str(fold_num)+ '_0.h5'
            model.load_weights(model_path_2)
            pred_1a = model.predict(test_images_2, batch_size=1, verbose=True)
            model_path_2 =  model_path+ 'CV_all_'+direction_2+'_'+ str(fold_num)+ '_1.h5'
            model.load_weights(model_path_2)
            pred_2a = model.predict(test_images_2, batch_size=1, verbose=True)
            model_path_2 =  model_path+ 'CV_all_'+direction_2+'_'+ str(fold_num)+ '_2.h5'
            model.load_weights(model_path_2)
            pred_3a = model.predict(test_images_2, batch_size=1, verbose=True)
            
            pred_a = (pred_1a+pred_2a+pred_3a)/3
            
            pred_1_post = post_processing(pred_c, ori_size_c, orient_c)
            pred_2_post = post_processing(pred_a, ori_size_a, orient_a)
            
            test_gt_1_post = post_processing(test_gt_1, ori_size_c, orient_c)
    
            pred = (pred_1_post+pred_2_post)/2#+pred_3
            pred[pred > 0.40] = 1.
            pred[pred <= 0.40] = 0.
            dsc = getDSC(pred, test_gt_1_post)
            h95 = hd95(pred, test_gt_1_post, voxelspacing=None, connectivity=1)
            vs = getVS(pred, test_gt_1_post) 
            print('Result of patient ' + str(ii))
            print('Dice (%)',                dsc,       '(higher is better, max=1)')
            print('H95 (mm)',                h95,       '(lower is better)')
            print('volume similarity (%)',   vs,        '(higher is better)')
            
            name = name_list[test_p_id]
            filename_resultImage = result_path+'pred_'+name[0:12]+'.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(pred), filename_resultImage )
    
            result_dsc.append(dsc*100)
            result_h95.append(h95)
            result_vs.append(vs)
            nameList.append(name[0:12])
result_dsc = np.round(result_dsc, 2)
result_h95 = np.round(result_h95, 2)
result_vs = np.round(result_vs, 2)
#    np.save('CV_all_dsc_scanner_'+str(ss)+'.npy', result_dsc)
#    np.save('CV_all_h95_scanner_'+str(ss)+'.npy', result_h95)
#    np.save('CV_all_vs_scanner_'+str(ss)+'.npy', result_vs)
#    np.save('CV_namelist_scanner_'+str(ss)+'.npy', nameList)
    
    


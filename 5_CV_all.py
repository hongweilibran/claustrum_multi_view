#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:31:36 2019

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
import warnings
K.set_image_data_format('channels_last')

smooth=1.

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

def get_unet(img_shape = None):
        inputs = Input(shape = img_shape)
        concat_axis = -1
        filters = 3
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
        model.compile(optimizer=Adam(lr=(1.5e-4)), loss=dice_coef_loss)

        return model

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

def train_leave_one_out(images, masks, direction, fold_num, iii, aug=False, verbose = True):
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if aug: model_path += 'Augmentation/'
    else: model_path += 'No_Augmentation/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path +=  'CV_all_'+direction+'_'+ str(fold_num)+ '_' + str(iii)+'.h5'
    
    if aug:
        images = np.concatenate((images, images[..., ::-1, :]), axis=0)
        masks = np.concatenate((masks, masks[..., ::-1, :]), axis=0)
    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    if aug: epoch = 1
    else: epoch = 240
    batch_size = 30

    img_shape = (row, col, 1)
    model = get_unet(img_shape)
  #  model.load_weights(model_path)
    current_epoch = 1
    while current_epoch <= epoch:
        print ('Epoch ', str(current_epoch), '/', str(epoch))
        if aug:
            images_aug = np.zeros(images.shape, dtype=np.float32)
            masks_aug = np.zeros(masks.shape, dtype=np.float32)
            for i in range(samples_num):
                images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
            image = np.concatenate((images, images_aug), axis=0)
            mask = np.concatenate((masks, masks_aug), axis=0)
        else:
            image = images.copy()
            mask = masks.copy()
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if history.history['loss'][-1] > 0.998:
            model = get_unet(img_shape)
            current_epoch = 1
        if history.history['loss'][-1] < 0.27:
            model.save_weights(model_path)
            break
    model.save_weights(model_path)
    print ('Model saved to ', model_path)

def main():
    warnings.filterwarnings("ignore")
#################
    os.environ["CUDA_VISIBLE_DEVICES"]="0"   
#    scanner = 1
    CV_list_all = [[3, 3, 3, 3, 3], [9, 9, 9, 9, 10], [20, 20, 21, 21, 21], [3, 3, 3, 4, 4]]
    direction = 'axial'
    direction = 'coronal'
#################    
    
    
    
    for fold_num in range(0, 5):
        for ss in range(1, 5):
            images = np.load('data_'+str(ss)+'_'+direction+'.npy')
            masks = np.load('gt_mask_'+str(ss)+'_'+direction+'.npy')
            slices_id = np.load('patient_id_'+str(ss)+'_'+direction+'.npy')
            CV_indices = np.load('scanner_'+str(ss)+'_'+'indices.npy')
            CV_list = CV_list_all[ss-1]
            if fold_num == 0:
                CV_indices_fold = CV_indices[0:CV_list[0]]
            else:
                CV_indices_fold = CV_indices[np.sum(CV_list[0:fold_num]): np.sum(CV_list[0:(fold_num+1)])]
            indices = np.sum((slices_id == CV_indices_fold), axis = 1)
            images_train = images[[not bool(x) for x in indices]]
            masks_train = masks[[not bool(x) for x in indices]]
            if ss == 1:
                imgs_temp = images_train
                masks_temp = masks_train
            else: 
                imgs_temp = np.concatenate((imgs_temp, images_train), axis = 0)
                masks_temp = np.concatenate((masks_temp, masks_train), axis = 0)
            
            
        for iii in range(0, 3):
            train_leave_one_out(imgs_temp, masks_temp, direction, fold_num, iii, aug=False, verbose = True)
    
if __name__=='__main__':
    main()


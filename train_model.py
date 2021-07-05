# for training a neonatal model 
# the hyperparameters are defined at the end of this file

import os
import argparse
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import SimpleITK as sitk
import scipy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
from PIL import Image
# for data augmentation
import imgaug as ia
import imgaug.augmenters as iaa
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

K.set_image_data_format('channels_last')
smooth=1.

########### data preparation
## pre-processing, crop or pad the volume with a reference size
def pre_processing(volume, per_, ref_size):
    rows, cols = ref_size[0], ref_size[1]
    dim_1 = np.shape(volume)[0]
    orig_rows, orig_cols = np.shape(volume)[1], np.shape(volume)[2]
    cropped_volume = []
    for nn in range(np.shape(volume)[0]):
        min_value = np.min(volume)
        if orig_rows >= rows and orig_cols >= cols:
            cropped_volume.append(volume[nn, int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows,
                                  int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols])
        elif orig_rows >= rows and cols >= orig_cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[:, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, 
                                                    int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows, :]
            cropped_volume.append(norm_slice)
        elif rows >= orig_rows and orig_cols >= cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, :] = volume[nn, :, int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols]
            cropped_volume.append(norm_slice)
        elif rows >= orig_rows and cols >= orig_cols:
            norm_slice = np.zeros((rows, cols))
            norm_slice[...] = min_value
            norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, :, :]
            cropped_volume.append(norm_slice)
    cropped_volume = np.asarray(cropped_volume)
    cropped_volume = cropped_volume[int(dim_1*per_): (dim_1 -int(dim_1*per_))]
    return cropped_volume[..., np.newaxis]

def inverse_orient(orient_):
    inv_orient = []
    if orient_ == [0, 1, 2]:
        inv_orient = (0, 1, 2)
    elif orient_ == [1, 0, 2]:
        inv_orient = (1, 0, 2)
    elif orient_ == [1, 2, 0]:
        inv_orient = (2, 0, 1)
    elif orient_ == [2, 1, 0]:
        inv_orient = (2, 1, 0)
    return inv_orient

## this is simply to check if the orientation of your images is right or not. Please check /images/coronal and /images/axial
def check_orient (array, direction, pat_count):
    for ss in range(0, np.shape(array)[0], 20):
        slice_ = 255*(array[ss] - np.min(array[ss]))/(np.max(array[ss]) - np.min(array[ss]))
        im = Image.fromarray(np.uint8(slice_))
        im.save(os.path.join(image_path, direction, str(pat_count)+'_'+str(ss)+'.png'))


def concatenate_arrays (all_array_1, array_2):
    if all_array_1 is None:
        all_array_1 = array_2
    else:
        all_array_1 = np.concatenate(([all_array_1 , array_2 ]), axis=0)
    return all_array_1

def preparation (full_path, pat_list): 
    image_con = None
    mask_con = None
    for pat_count, pat in enumerate(sorted(pat_list)):
        print('preparation of subject: ', pat_count)
        #read data, please name the segmentation mask with the keword 'seg'
        pat_file_name = glob.glob(os.path.join(full_path, pat, '*T2w.nii.gz'))
        seg_file_name = glob.glob(os.path.join(full_path, pat, '*seg*')) 
        if len(pat_file_name)==0 or len(seg_file_name)==0:
            print('ignoring', pat)
            continue
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(pat_file_name[0]))  
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_file_name[0]))  

        # z-score normalization of the image in a scan level
        brain_mask_T2 = np.zeros(np.shape(image_array), dtype = 'float32')
        brain_mask_T2[image_array >=thresh] = 1
        brain_mask_T2[image_array < thresh] = 0
        for iii in range(np.shape(image_array)[0]):
            brain_mask_T2[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T2[iii,:,:])  #fill the holes inside br
        image_array = image_array - np.mean(image_array[brain_mask_T2 == 1])
        image_array /= np.std(image_array[brain_mask_T2 == 1])

        # label adaption: label 1 and 2 become label 1
        brain_mask_T2 = np.zeros(np.shape(mask_array), dtype = 'float32')
        brain_mask_T2[mask_array >=1] = 1
        brain_mask_T2[mask_array < 1] = 0
        for iii in range(np.shape(mask_array)[0]):
            brain_mask_T2[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T2[iii,:,:])  #fill the holes inside br
        mask_array = brain_mask_T2

        # transform/project the original array to axial and coronal views
        if args.orientation == 'coronal':
            image_array = np.transpose(image_array, (1, 0, 2))
            print(np.shape(image_array))
            mask_array = np.transpose(mask_array, (1, 0, 2))
            # this is to check the orientation of your images is right or not. Please check /images/coronal and /images/axial
            check_orient (image_array, direction_1, pat_count)
        elif args.orientation == 'axial':
            image_array = image_array  
            print(np.shape(image_array))
                # this is to check the orientation of your images is right or not. Please check /images/coronal and /images/axial  
            check_orient (image_array, direction_2, pat_count) 
        else:
            print('--> error: undefined orientation: ', orientation)
            exit()
            
        # pre-processing, crop or pad them to a standard size given by img_shape    
        image_array = pre_processing(image_array, per, img_shape)
        mask_array = pre_processing(mask_array, per, img_shape)


        # array concatenation
        image_con = concatenate_arrays(image_con, image_array)
        mask_con = concatenate_arrays(mask_con, mask_array)

    return image_con, mask_con


# image augmentation with scale, shift, rotate and shear
seq = iaa.Sequential([iaa.Affine(scale={"x":(0.9, 1.1), "y":(0.9, 1.1)}, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-15, 15), shear=(-4, 4))], random_order=True) 

def image_aug(image_array, mask_array):
    input_array = np.concatenate((image_array, mask_array), axis = 3)
    aug_array = []
    for ii in range(np.shape(input_array)[0]):
        aug_out = seq(images=input_array[ii:ii+1])
        aug_array.append(aug_out[0])

    all_array = np.concatenate((input_array, np.asarray(aug_array)),axis = 0)
    image_array_new = all_array[..., 0:1]  
    mask_array_new = all_array[..., 1:2]
    return image_array_new, mask_array_new


############## training

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

## a component of U-Net, to ensure the sizes between skip connections are matched
def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw//2), int(cw//2) + 1
        else:
            cw1, cw2 = int(cw//2), int(cw//2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch//2), int(ch//2) + 1
        else:
            ch1, ch2 = int(ch//2), int(ch//2)

        return (ch1, ch2), (cw1, cw2)

## the network architecture
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


## data preparation and training of the model with the given parameters
def prep_train(csv_logger, args): 
    global model
    global pretrained_model
    image_list = sorted(os.listdir(train_image_path))
    image_array, mask_array = preparation (train_image_path, image_list)

    # for data augmentation:
    image_array, mask_array = image_aug(image_array, mask_array)

    # The while-loop assures that the model does not get stuck in the first epoch. 
    # If the model does not learn in one epoch, it is rare that it learns in following epochs. 
    # Thus, the weights are reloaded and the training starts again.
    current_epoch = 0
    while (current_epoch < args.n_epochs):
        print('epoch: '+str(current_epoch))
        history = model.fit(image_array, mask_array, batch_size = args.batch_size, epochs = 1, callbacks=[csv_logger]) 
        current_epoch +=1
        if history.history['loss'][0] > 0.998: 
            print('loss problem')
            model.load_weights(pretrained_model)
            # for training from scratch: comment the line above and uncomment the line below
            #model = get_unet(img_shape)
            current_epoch = 0  
        else: 
            model.fit(image_array, mask_array, batch_size = args.batch_size, epochs = 1, callbacks=[csv_logger])
        if current_epoch% 4 ==0: # save the model every 4 epochs
            model.save_weights(save_model_path+args.name_exp+'_'+args.orientation+'_model'+'_epoch'+str(args.n_epochs)+'.h5')

per = 0.25
thresh = 10

direction_1 = 'coronal'
direction_2 = 'axial'

model_path = 'models/'
save_model_path = 'saved_models/'
train_image_path = 'data/training/'
image_path = 'images/'

img_shape = (200, 200, 1)


### hints to train a new model:
# change the name_exp for each new model to prevet overwriting the previous one
# you might adapt the loaded model weights for transfer learning

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_exp","--name_exp", type=str, default='initial', help="define the name of your experimetn")
    parser.add_argument("-b","--batch_size", type=int, default=10, help="define the batch size")
    parser.add_argument("-n","--n_epochs", type=int, default=5, help="define the number of epochs")
    parser.add_argument("-o","--orientation", type=str, help="axial or coronal, the orientation property of the sub-network")
    args = parser.parse_args()
    
    if args.orientation != 'axial' or args.orientation != 'coronal':
        assert("wrong input for the orientation.")

    model = get_unet(img_shape)
    pretrained_model = os.path.join(model_path, 'axial_0.h5') # change the pretrained model if you want
    # load pre-trained models available. For training from scratch you can comment the next line
    model.load_weights(pretrained_model)
    
    ## to save the training loss in a csv file
    logger_name = os.path.join('scores/'+'model'+str(args.name_exp)+'_log.csv')
    csv_logger = CSVLogger(logger_name, append=True, separator=',')

    model.summary()

    prep_train(csv_logger, args)

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras import backend as K
import SimpleITK as sitk
import scipy
from PIL import Image

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

## simple post-processing
def post_processing(input_array, ori_size, orient_1):
   # output_array = np.zeros(ori_size, dtype= 'float32')
    output_array = crop_or_pad(input_array, per, ori_size)
    inv_orient = inverse_orient(orient_1)
    output_array = np.transpose(output_array, inv_orient)
    return output_array


direction_1 = 'coronal'
direction_2 = 'axial'

per = 0.2
thresh = 10
img_shape = (180, 180, 1)
model = get_unet(img_shape)
model_path = 'models/'
result_path = 'output/'
data_path = 'data/test/'
image_path = 'images'

if not os.path.exists(result_path):
    os.mkdir(result_path)

pat_list = os.listdir(data_path)

for pat in pat_list:
    #read data
    pat_file_name = os.path.join(data_path, pat, pat+'_sMRI_denoised.nii.gz')
    ref_image = sitk.ReadImage(pat_file_name)
    image_array = sitk.GetArrayFromImage(ref_image)
    
    if not os.path.exists(os.path.join(result_path, pat)):
        os.mkdir(os.path.join(result_path, pat))
    
    # z-score normalization
    brain_mask_T1 = np.zeros(np.shape(image_array), dtype = 'float32')
    brain_mask_T1[image_array >=thresh] = 1
    brain_mask_T1[image_array < thresh] = 0
    for iii in range(np.shape(image_array)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside br
    image_array = image_array - np.mean(image_array[brain_mask_T1 == 1])
    image_array /= np.std(image_array[brain_mask_T1 == 1])
    
    # transform/project the original array to axial and coronal views
    corona_array = np.transpose(image_array, (1, 0, 2))
    axial_array = image_array
     
    #this is to check the orientation of your images is right or not. Please check /images/coronal and /images/axial
    for ss in range(np.shape(corona_array)[0]):
        slice_ = 255*(corona_array[ss] - np.min(corona_array[ss]))/(np.max(corona_array[ss]) - np.min(corona_array[ss]))
        im = Image.fromarray(np.uint8(slice_))
        im.save(os.path.join(image_path, direction_1, str(ss)+'.png'))
    
    for ss in range(np.shape(axial_array)[0]):
        slice_ = 255*(axial_array[ss] - np.min(axial_array[ss]))/(np.max(axial_array[ss]) - np.min(axial_array[ss]))
        im = Image.fromarray(np.uint8(slice_))
        im.save(os.path.join(image_path, direction_2, str(ss)+'.png')) 
    
    #original size and the orientations 
    ori_size_c = np.asarray(np.shape(corona_array))
    ori_size_a = np.asarray(np.shape(axial_array))
    orient_c = [1, 0, 2]
    orient_a = [0, 1, 2]
    
    #pre-processing, crop or pad them to a standard size [N, 180, 180]
    corona_array =  pre_processing(corona_array, per, img_shape)
    axial_array =  pre_processing(axial_array, per, img_shape)

    #do inference on different views 
    model_path_1 = os.path.join(model_path,direction_1+'_0.h5')
    model.load_weights(model_path_1)  
    pred_1c = model.predict(corona_array, batch_size=1, verbose=True)
    model_path_2 = os.path.join(model_path,direction_1+'_1.h5')
    model.load_weights(model_path_2)  
    pred_2c = model.predict(corona_array, batch_size=1, verbose=True)
    model_path_3 = os.path.join(model_path,direction_1+'_2.h5')
    model.load_weights(model_path_3)  
    pred_3c = model.predict(corona_array, batch_size=1, verbose=True)
    # ensemble 
    pred_c = (pred_1c+pred_2c+pred_3c)/3
    
    model_path_1 = os.path.join(model_path,direction_2+'_0.h5')
    model.load_weights(model_path_1)  
    pred_1a = model.predict(axial_array, batch_size=1, verbose=True)
    model_path_2 = os.path.join(model_path,direction_2+'_1.h5')
    model.load_weights(model_path_2)  
    pred_2a = model.predict(axial_array, batch_size=1, verbose=True)
    model_path_3 = os.path.join(model_path,direction_2+'_2.h5')
    model.load_weights(model_path_3)  
    pred_3a = model.predict(axial_array, batch_size=1, verbose=True)
    
    pred_a = (pred_1a+pred_2a+pred_3a)/3
    
    # transform them to their original size and orientations
    pred_1_post = post_processing(pred_c, ori_size_c, orient_c)
    pred_2_post = post_processing(pred_a, ori_size_a, orient_a)
    
    # ensemble of two views
    pred = (pred_1_post+pred_2_post)/2
    pred[pred > 0.40] = 1.
    pred[pred <= 0.40] = 0.
    
    #save the masks
    sitk_image = sitk.GetImageFromArray(pred)
    sitk_image.CopyInformation(ref_image)
    filename_resultImage = os.path.join(result_path, pat, 'pred_mask.nii.gz')
    sitk.WriteImage(sitk_image, filename_resultImage )



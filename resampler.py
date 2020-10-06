import nibabel as nib
from dipy.align.reslice import reslice
import os


def resample(image_path, output_path=None, new_spacing=(1, 1, 1), order=3):
    if output_path is None:
        output_path = image_path
    im = nib.load(image_path)
    # get necessary info.
    header = im.header
    vox_zooms = header.get_zooms()
    vox_arr = im.get_fdata()
    vox_affine = im.affine
    # resample using DIPY.ALIGN
    if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        new_spacing = (new_spacing[0], new_spacing[1], new_spacing[2])
    new_vox_arr, new_vox_affine = reslice(vox_arr, vox_affine, vox_zooms, new_spacing, order=order)
    # create reoriented NIB image
    new_im = nib.Nifti1Image(new_vox_arr, new_vox_affine, header)
    nib.save(new_im, output_path)
    print('Image \"' + image_path + '\" resampled and saved to \"' + output_path + '\".')



import os
import glob
import cv2
import imageio

import numpy as np 
import pandas as pd 
import nibabel as nib
import matplotlib.pyplot as plt

def show_mid_slice(img_numpy, title='img'):

   """
   Accepts an 3D numpy array and shows median slices in all three planes
   """
   assert img_numpy.ndim == 3
   n_i, n_j, n_k = img_numpy.shape

   # sagittal (left image)
   center_i1 = int((n_i - 1) / 2)
   # coronal (center image)
   center_j1 = int((n_j - 1) / 2)
   # axial slice (right image)
   center_k1 = int((n_k - 1) / 2)

   show_slices([img_numpy[center_i1, :, :],
                img_numpy[:, center_j1, :],
                img_numpy[:, :, center_k1]])
   plt.suptitle(title)

def show_slices(slices):
   """
   Function to display a row of image slices
   Input is a list of numpy 2D image slices
   """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
        # why must transpose
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
   plt.savefig("fig.png")


nii_image=nib.load("/root/repo/liver-tumor-segmentation/data/volume-0.nii")
# data= nii_image.get_fdata()
# show_mid_slice(data, 'first image')


header = nii_image.header
# Extract the "srow_z" values
srow_z = header.get_best_affine()[2]
srow_z1=header['srow_z'][-2]
# print (header['srow_z'])
 #header.get_best_affine()[2]  is equivalent to header['srow_z']
# print ("1",srow_z1)
# print("2", srow_z)
# print ("affine matrix ",header.get_best_affine())
# print("Affine transformation matrix:\n", nii_image.affine)

import os
import numpy as np 
import nibabel as nib
from scipy import ndimage


class Param():
    '''
    parameter class to store all the parameters
    '''
    def __init__(self, resize_option = "by_zdist"):
        self.window_min = -100
        self.window_max = 400
        self.patch_shape = (128, 128, 16)  # used for resizing by slice spacing
        self.equalize_histogram = False  
        self.normalize = True
        self.resize_option = resize_option  # options are "by_zdist" or "by_vol"
        self.zoom_order = 3
        if self.resize_option == "by_zdist":
            self.zdist = 2  # set z spacing to zdist mm, only vlid when resize option is by zdist
        elif self.resize_option == "by_vol":
            self.resized_vol_shape = (128, 128, 128)  # used for resizing volume to certain shape
        else:
            raise ValueError(f"{self.resize_option} is not a valid resize option")
        self.output_type = 'npy' 


# preprocessing functions
def read_nii(f):
    img_obj = nib.load(f)
    img_data = img_obj.get_fdata()
    
    return img_data, img_obj.header

def hist_eq(image, number_bins=32):
    # histogram equalization
    # adopt from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)
# why -100 and 400
def windowing(nparray_2d, _min, _max):
    # Setting hounsfield unit values to [−100, 400] to discard irrelevant structures
    np.clip(nparray_2d, _min, _max)

def norm(nparray):
    # normalize scans to [0,1]
    _min = nparray.min()
    _max = nparray.max()
    nparray = nparray - _min
    nparray = nparray / (_max - _min)
    return nparray

def norm_zscore(nparray):
    # normalize 2d scands by mean and standard deviation
    mean = nparray.mean()
    std = nparray.std()    
    nparray = nparray - mean
    nparray /= std
    return nparray

def resize_volume(orig_volume, zdist, params, order):
    """
    resize orig_volume to desired dimension defined in parameters
    zdist: zdist of the volume
    """
    resize_factor = [0]*3
    if params.resize_option == "by_zdist":
        for i in range(2):
            resize_factor[i] = params.patch_shape[i]/orig_volume.shape[i]
        # rescale scan spacing to 2mm
        resize_factor[2] = zdist/params.zdist
    elif params.resize_option == "by_vol":
        for i in range(3):
            resize_factor[i] = params.resized_vol_shape[i]/orig_volume.shape[i]
    else:
        raise ValueError(f"{params.resize_option} is not a valid resize option")
    print ("factor",resize_factor)
    resized_vol = ndimage.zoom(orig_volume, resize_factor, order = order)
    return resized_vol

def upsample_volume(downsampled_vol, original_vol_shape, order):
    resize_factor = [0]*3
    for i in range(3):
        resize_factor[i] = original_vol_shape[i]/downsampled_vol.shape[i]
    return ndimage.zoom(downsampled_vol, resize_factor, order = order)

def preprocessing_vol(f_vol, param):
   
    vol, header = read_nii(f_vol)
    print ("before process vol",vol.shape)
    zdist = header['srow_z'][-2]
    # windowing
    vol = np.clip(vol, param.window_min, param.window_max)  
    # resizing vol
    vol = resize_volume(vol, zdist, param, order = param.zoom_order)
    
    # histogram equalization
    if param.equalize_histogram:
        for i in range(vol.shape[-1]):
            vol[:,:,i] = hist_eq(vol[:,:,i])
    
    # normalizing
    if param.normalize:
        vol = norm(vol)
#        vol = norm_zscore(vol)
        
    # output zdist for preprocessing_mask (spacing between scan and mask is different for some cases)
    return vol, zdist  

def preprocessing_mask(f_mask, zdist, param):
    mask, _ = read_nii(f_mask)
    mask = resize_volume(mask, zdist, param, order = param.zoom_order)
    # what is np.rint
    # np.rint() function is a NumPy function that stands 
    # for "round to the nearest integer."
    # before np.rint, the mask value is extremely small float
    # print ("before",mask)
    mask = np.rint(mask)
    # print ("after",mask)
    return mask.astype(int)
param =Param()
f_vol="/root/repo/liver-tumor-segmentation/data/volume-0.nii.gz"
f_mask="/root/repo/liver-tumor-segmentation/data/segmentation-0.nii.gz"
vol, zdist = preprocessing_vol(f_vol, param) 
print ("volshape", vol.shape)
print ("z",zdist)
mask = preprocessing_mask(f_mask, zdist, param)
print ("maskshape", mask.shape)

#before 512,512,75
# zdist=5
#after 128,128,88
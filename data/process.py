import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import sys
sys.path.append("/root/repo/liver-tumor-segmentation/")
import argparser
import nibabel as nib

class LITS_preprocess:
    def __init__(self, raw_dataset_path,fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  
        self.size = args.min_slices  
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate

    #order :
    # Trim the original image and save it."
    # tool.write_train_val_name_list()  
    def fix_data(self):
        # does this fix data method and process method apply to test set
        if not os.path.exists(self.fixed_path):    
            os.makedirs(join(self.fixed_path,'ct'))
            os.makedirs(join(self.fixed_path, 'label'))
        file_list = os.listdir(join(self.raw_root_path,'ct'))
        Numbers = len(file_list)
        print('Total numbers of samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(self.raw_root_path, 'ct', ct_file)
            seg_path = os.path.join(self.raw_root_path, 'label', ct_file.replace('volume', 'segmentation'))
            new_ct, new_seg = self.process(ct_path, seg_path, classes = self.classes)
            # new_ct, new_seg are new sitk object
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))  
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label', ct_file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))

    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)

        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:",ct_array.shape, seg_array.shape)
        # Ori shape: (75, 512, 512) (75, 512, 512)
        if classes==2:
            # Merge the liver and liver tumor labels in the ground truth into one label."
            seg_array[seg_array > 0] = 1
        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        # order=3' specifies cubic interpolation,
        #self.slice_down_scale, 1.0
        #self.xy_down_scale, 0.5
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=3)
        #order=0, it indicates that the interpolation method used is "nearest-neighbor" 
        #ct_array.shape # (375, 256, 256), 375=75*5
        # why scale along z axis like this
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=0)
        # does this zoom affect image 
        # np.unique(seg_array)) : [0,1,2]: 0, background, 1, liver, 2: tumor 
        # Find the starting and ending slices of the liver region and expand them in both directions."
        z = np.any(seg_array, axis=(1, 2)) #axis = plane xy
        # print ("z",z.shape) # (375,)
        
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        # numpy.where(condition, [x, y, ]/)
        #Return elements chosen from x or y depending on condition.#
     # np where return a tuple , if not specify more,the first one is the list of index in original array
     # where condition is True, the second onece is False, here [0] is for get the liver region
     # [[0, -1]] is get the first and last xy plane 
        #"Expand in each direction by a certain number of voxels.
        # what is self.endplacd slice
        # self.expand_slice=20
        # this expand slice in both start and end slice
        if start_slice - self.expand_slice < 0:
            
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += self.expand_slice

        print("Cut out range:",str(start_slice) + '--' + str(end_slice))
         # "If the remaining number of slices is insufficient to reach the desired 
         #size, discard the data. As a result, there will be very few instances of data."
        # self.size is probably theminimum number of slices
        if end_slice - start_slice + 1 < self.size:
           
            print('Too little slice，give up the sample:', ct_file)
            return None,None
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:",ct_array.shape,seg_array.shape)
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        # why origin is unchanged
        new_ct.SetOrigin(ct.GetOrigin())
        # hay, this is set the space to reverse of  zoom above
         #self.slice_down_scale, 1.0
        #self.xy_down_scale, 0.5
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        # 
        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        
        return new_ct, new_seg

    def write_train_val_name_list(self):
        data_name_list = os.listdir(join(self.fixed_path, "ct"))
        data_num = len(data_name_list)
        random.shuffle(data_name_list)
        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
        val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, "train_path_list.txt")
        self.write_name_list(val_name_list, "val_path_list.txt")


    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'segmentation'))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()

if __name__ == '__main__':
    raw_dataset_path = '/root/data/liver/train'
    fixed_dataset_path = '/root/data/liver/fix_train'
 # does nitk and nib read differently ?
    args = argparser.args 
    if not os.listdir(fixed_dataset_path):
        tool = LITS_preprocess(raw_dataset_path,fixed_dataset_path, args)
        tool.fix_data()                            
        tool.write_train_val_name_list()  
    f= "/root/data/liver/fix_train/ct/volume-5.nii"
    ct = sitk.ReadImage(f, sitk.sitkInt16)
    arr=sitk.GetArrayFromImage(ct)
    # print (ct.GetSpacing())


    # nii_image=nib.load("/root/repo/liver-tumor-segmentation/data/volume-0.nii")
    # tool.process(ct1,seg1,classes=3)
    # header = nii_image.header
    # note that each image has differnet affine matrix
    # print (header.get_best_affine(),)
    # matrix 
#     [[  -0.703125     0.           0.         172.8999939]
#  [   0.           0.703125     0.        -179.296875 ]
#  [   0.           0.           5.        -368.       ]
#  [   0.           0.           0.           1.       ]]
    # print (header['srow_z'])
    #header.get_best_affine()[2]  is equivalent to header['srow_z']

       

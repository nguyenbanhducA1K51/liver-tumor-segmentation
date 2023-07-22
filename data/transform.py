from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

class Resize:
    def __init__(self, scale):
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        self.scale = scale

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=(1,self.scale,self.scale),mode='trilinear', align_corners=False, recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)
        return img[0], mask[0]
class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
class RandomCrop:
    def __init__(self, slices):
        # desired crop slices
        self.slices =  slices

    def _get_range(self, slices, crop_slices):
        # slices is number of slices in the image
        # crop_slices is the number of slices after crop
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):

        ss, es = self._get_range(mask.size(1), self.slices)
        # ss is start slice, es is end  slice
        # print(self.shape, img.shape, mask.shape)
        # actually the first dim is obtain by unsqueeze(0)
        tmp_img = torch.zeros((img.size(0), self.slices, img.size(2), img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), self.slices, mask.size(2), mask.size(3)))
        tmp_img[:,:es-ss] = img[:,ss:es]
        tmp_mask[:,:es-ss] = mask[:,ss:es]
        return tmp_img, tmp_mask
class Center_Crop:
    def __init__(self, base=16 max_size=96):
        # default: base :16, max_size:96
        # "The 'base' is set to 16 by default, because after four downsamplings, it becomes 1."
        self.base = base  
        self.max_size = max_size 
        if self.max_size%self.base:
            #max_size limits the maximum number of sampled slices to prevent 
            #GPU memory overflow and should also be a multiple of 16."
        
            self.max_size = self.max_size - self.max_size%self.base 
    def __call__(self, img , label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1)//2 - slice_num//2
        right =  img.size(1)//2 + slice_num//2

        crop_img = img[:,left:right]
        crop_label = label[:,left:right]
        return crop_img, crop_label


from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *
from fastai.medical.imaging import *

import pydicom,kornia,skimage
from pydicom import dcmread
# fast ai seem to also have method dcmread, which is 
#TEST_DCM = Path('images/sample.dcm')
#dcm = TEST_DCM.dcmread()

from fastai.medical.imaging import PILDicom
import torch
# look at this page  https://docs.fast.ai/data.external.html
pneumothorax_source = untar_data(URLs.SIIM_SMALL)
items = get_dicom_files(pneumothorax_source, recurse=True, folders='train')

img = items[10]
dimg = dcmread(img)
#pydicom.dataset.FileDataset
# to get the np array, use attribute pixel_array, dimg.pixel_array.shape

# Class TensorDicom
# Inherits from TensorImage and converts the pixel_array into a TensorDicom
class TensorDicom(TensorImage): _show_args = {'cmap':'bone'}
ten_img = TensorDicom(dimg.pixel_array)
# ten_img.show()
test_i="/root/repo/liver-seg/13.dcm"
test_im = dcmread(test_i)
timg = PILDicom.create(test_i)
# print (type(timg))
# fastai.medical.imaging.PILDicom
timg.show()
plt.savefig("liver.png")
plt.clf()


@patch(as_prop=True)
# ! important, normally if there no condition as_prop=True, the method below 
# will be called as normal method. Else, it will call in attribute manner (without the ())
def scaled_px(self:DcmDataset):
    # DcmDataset aka pydicom.dataset.FileDataset
    "`pixels` scaled by `RescaleSlope` and `RescaleIntercept`"
    img = self.pixels
    if hasattr(self, 'RescaleSlope') and hasattr(self, 'RescaleIntercept') is not None:
        print ("has rescale")
        print (self.RescaleSlope,self.RescaleIntercept )
        return img * self.RescaleSlope + self.RescaleIntercept 
    else: return img

# plt.hist(test_im.pixels.flatten(), color='c')
# plt.show()
# plt.savefig('dee.png')
# plt.clf()

# print ("test im",type(test_im))
tensor_dicom_scaled = test_im.scaled_px#convert into tensor taking RescaleIntercept and RescaleSlope into consideration
# plt.hist(tensor_dicom_scaled.flatten(), color='c',bins=50)
# plt.show()
# plt.savefig('scaled_dee.png')


#Air has a value of -1000 Hounsfield Units(HUs),
# water has a value of 0 HUs and depending on the type of bone, bone has values between 300 and 2000 HUs
# why in this image has min -2000


@patch
def freqhist_bins(self:Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()

@patch
def hist_scaled_pt(self:Tensor, brks=None):
    # Pytorch-only version - switch to this if/when interp_1d can be optimized
    if brks is None: brks = self.freqhist_bins()
    brks = brks.to(self.device)
    ys = torch.linspace(0., 1., len(brks)).to(self.device)
    return self.flatten().interp_1d(brks, ys).reshape(self.shape).clamp(0.,1.)

t= test_im.scaled_px.freqhist_bins()


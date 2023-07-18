# https://levelup.gitconnected.com/automatic-liver-segmentation-part-2-4-data-preparation-and-preprocess-981d790f5555
import dicom2nifti
import os
import glob
import nibabel as nib
import numpy as np
def create_group(in_dir,out_dir,Number_slices):
    # this funciton is to get the last part of the path so that we can use it to name the folder
    # 'in_dir': the path to your folder that contain dicom files
    # 'out_dir': the path where you want to put the converted nifti files
    # 'Number_slices': here you put the number of clisesthat you need for your project an dit will create griup with this number
    for patient in glob.glob(in_dir+'/*'):
        # os.path.normpath normalize by remove reduntdant character
        #os.path.basename get the tail .e.g
        # os.path.basename ( /root/patient)= patient
        # os.path.basename ( /root/patient.py)= patient.py
        patient_name=os.path.basename(os.path.normpath(patient))
        # calculatye numebr of folders
        number_folders=int(len(glob(patient+'/*'))/Number_slices)
        for i in range(num_folders):
            output_path=os.path.join(out_dir,patient_name+'_'+str(i))
            os.mkdir(output_path)
            for i, file  in enumerate (glob.glob(patient+'/*')):
                if i==Number_slices+1:
                    break
                    shutil.mov(file,output_path)
def dcm2nifti(in_dir,out_dir):
    for folder in tqdm(glob.glob(in_dir+'/*')):
        patient_name=os.path.basename(os.path.normpath(folder))
        dicom2nifti.dicom_series_to_nifti(folder,os.path.join(out_dir,patient_name+'.nii.gz'))

def find_emty(in_dir):
    # this method help find empty volumns that may not need for training
    list_patients=[]
    for patient in glob.glob(os.path.join(in_dir,'*')):

# img = nib.load('path/to/your/file.nii.gz')

# # Access the image data and other metadata
# data = img.get_fdata()
#img.get_fdata(): This retrieves the image data as a NumPy array
# header = img.header
# affine = img.affine
        img=nib.load(patient)
        # what is this ?
        # If the image data has more than 2 unique values, 
        #it implies that the volume contains information beyond just empty or background values.
        if len(np.unique(img.get_fdata()))>2:
            print (os.path.basename(os.path.normpath(patient)))
            list_patients.append(os.path.basename(os.path.normpath(patient)))
    return list_patients
from .fract import fract
import os
import nibabel as nib

def run_fract(subjid, image, output_folder=None):
    if output_folder is None:
        print ("The output_folder is the current directory")
        output_folder = os.getcwd()
    else:
        print ("The output_folder is: ", output_folder)
        
    if isinstance(image, nib.Nifti1Image):
        fract(subjid, image, output_folder)
    elif isinstance(image, str) and (image.endswith('.nii') or image.endswith('.nii.gz')):
        fract(subjid, image, output_folder)
    else:
        print("ERROR: The passed image should be nifti file or nifti file path")

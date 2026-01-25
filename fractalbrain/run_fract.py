from .fract import fract
import os
import nibabel as nib

def run_fract(subjid, image, output_folder=None, scaling_method="exponential", save_plots=True):
    if output_folder is None:
        print ("The output_folder is the current directory")
        output_folder = os.getcwd()
    else:
        print ("The output_folder is: ", output_folder)
        
    if isinstance(image, nib.Nifti1Image):
        return fract(subjid, image, output_folder, scaling_method, save_plots)
    elif isinstance(image, str) and (image.endswith('.nii') or image.endswith('.nii.gz')):
        return fract(subjid, image, output_folder, scaling_method, save_plots)
    else:
        print("ERROR: The passed image should be nifti file or nifti file path")

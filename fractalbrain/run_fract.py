from .fract import fract
import os

def run_fract(subjid, image, output_folder=None):
    imagefile = os.path.basename(image)
    imagename, image_extension1 = os.path.splitext(imagefile)
    imagename, image_extension2 = os.path.splitext(imagename)
    image_extension = image_extension2 + image_extension1

    if image_extension == '.nii' or image_extension == '.nii.gz':
        print ("The prefix is: ", subjid)
        print ("The NifTI image is: ", image)
        if output_folder is None:
          print ("The output_folder is the current directory")
        else:
          print ("The output_folder is: ", output_folder)
      
        fract(subjid, image, output_folder)

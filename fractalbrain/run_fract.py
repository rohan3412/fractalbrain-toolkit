from fractalbrain.fract import fract

def run_fract(subjid, image, output_folder=None):
    if image_extension == '.nii' or image_extension == '.nii.gz':
        print ("The prefix is: ", subjid)
        print ("The NifTI image is: ", image)
        if output_folder is None:
          print ("The output_folder is the current directory")
        else:
          print ("The output_folder is: ", output_folder)
      
        fract(subjid, image, output_folder)

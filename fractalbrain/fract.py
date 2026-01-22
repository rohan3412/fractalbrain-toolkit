from fractalbrain.asofi import asofi

def fract(subjid, image, output_folder=None,scaling_method="exponential"):
    print(f"Starting fract for {subjid}\n")
    return asofi(subjid, image, output_folder,scaling_method)
    
            
    
    
    
    
    
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:55:49 2019

@author: Chiara Marzi, Ph.D. student in Biomedical, Electrical and System Engineering,
at Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
University of Bologna, Bologna, Italy. 
E-mail address: chiara.marzi3@unibo.it

fractalbrain toolkit e-mail address: fractalbraintoolkit@gmail.com
"""

def fract( subjid, image, output_folder = None ):
    from fractalbrain.asofi import asofi
    import logging
    import os
    import time
    import datetime

    ### START TIME ###
    start_time = time.process_time()
    start_time_to_log = time.asctime( time.localtime(time.time()) )
    NOW = datetime.datetime.now()
    DATE = NOW.strftime("%Y-%m-%d")
    TIME = NOW.strftime("%H-%M-%S")
    
    ### LOG FILE SETTING ###
    imagepath = os.path.dirname(image)
    if not imagepath or imagepath == '.':
        imagepath = os.getcwd()

    if output_folder is None:
        log_file_name = output_folder + '/' + subjid +'/'+subjid+'_fractal_'+DATE+'_'+TIME
    else:
        subject_folder = os.path.join(output_folder, subjid)
        
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
            
        log_file_name = os.path.join(subject_folder, subjid+'_fractal_'+DATE+'_'+TIME)
        
    log = logging.getLogger(log_file_name)
    hdlr = logging.FileHandler(log_file_name+'.log', mode="w")
    formatter = logging.Formatter(fmt = '%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S') 
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr) 
    log.setLevel(logging.INFO)
    
    log.info('Started at %s', start_time_to_log)
    
    ### FRACTAL ANALYSIS ###
    asofi(subjid, image, output_folder)
    
    ### END TIME ###
    end_time = time.process_time()
    end_time_to_log = time.asctime( time.localtime(time.time()) )
    elapsed_time = end_time - start_time
    log.info('#----------------------------------------')
    log.info('Started at %s', start_time_to_log)
    log.info('Ended at %s', end_time_to_log)
    log.info('fract-run-time-seconds %s', elapsed_time)
    return;
            
    
    
    
    
    
    


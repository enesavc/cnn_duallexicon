#!/usr/bin/env python
# coding: utf-8

# # Check GPU

# In[1]:


#Check for GPU
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_logical_devices('GPU')))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None)


# # tfcochleagram Generation
# 
# Loads an audio file and generates a cochleagram using the tfcochleagram.py library. 
# 
# 

# In[2]:


from __future__ import division
# For displaying audio and images in notebook
#import IPython.display as ipd

from PIL import Image
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import scipy
import time
import os
from os import listdir
from os.path import isdir, join, dirname, join, realpath
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tfcochleagram
import h5py

# Helper functions for loading audio
from utils import *


# In[6]:


t0 = time.time()
cwd = os.getcwd()
PATH = join(cwd, 'input')
#print(PATH)

onlydirs = [f for f in listdir(PATH) if isdir(join(PATH, f))]
j = 0
for dirs in onlydirs:
    dirname = PATH + "/" + dirs 
    rfnArray = [os.path.join(dirname, f)for f in os.listdir(dirname) if f.endswith('.wav')]
    i = 0
    
    for f in rfnArray:
        j = j + 1
        print(j, end='\r')
        #print('Running demo with sound file: %s ' % f)
        test_audio, SR = load_audio_wav_resample(f, DUR_SECS='full', resample_SR=20000)
        # Generally a good idea to rms normalize the audio
        test_audio = rms_normalize_audio(test_audio, rms_value=0.01)
        # Using rFFT below, and it is currently implemented only for even # signals. 
        if len(test_audio.ravel())%2:
            test_audio = test_audio[:-1]
            #print(test_audio)
        if len(test_audio.shape) == 1: # we need to make sure the input node has a first dimension that corresponds to the batch size
            test_audio = np.expand_dims(test_audio,0) 
        nets = {}
        # tfcochleagram expects a dictionary with 'input_signal' defined for the input audio
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets = tfcochleagram.cochleagram_graph(nets, SR, rFFT=True)
        #with tf.Session() as sess:
        #with tf.compat.v1.Session() as sess:
        #nets['input_signal'] = test_audio
        #cochleagram = nets['cochleagram']
        #filters_out = nets['filts_tensor']
        
        #save the cochs into pngs
        MAIN_PATH = join(cwd, 'output')
        if isdir(MAIN_PATH + '/%s' %dirs) == False:      
         os.mkdir(MAIN_PATH + '/%s' %dirs)
        
        filenames_with_extension = os.listdir(dirname)
        
        filenames=[x.split('.wav')[0] for x in filenames_with_extension]
        filename=filenames[i]
        i += 1
        #filex=os.path.basename(filename)       

        #write to png
        import matplotlib
        matplotlib.image.imsave(MAIN_PATH + '/%s' %dirs + '/' + filename +'.png', nets['cochleagram'][0,:,:,0], origin='lower', cmap='Blues')
       
        
print(nets['cochleagram'].shape)
t1 = time.time()
timer = t1-t0
print(timer)
#plt.matshow(cochleagram[0,:,:,0], origin='lower', cmap='Blues')
#plt.colorbar()


# In[15]:


#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize) #to see the numpy array completely
#numpy.set_printoptions(threshold=False) #back to normal
#a = nets['cochleagram'][0,:,:,0]
#numpy.savetxt("foo.csv", a, delimiter=",") #save the cochleagram to csv


# In[ ]:





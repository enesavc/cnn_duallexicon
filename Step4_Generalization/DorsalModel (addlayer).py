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


# ## Set Up Model Parameters

# In[2]:


import numpy as np
import pandas as pd
import keras
from keras.models import Model,load_model
from keras import backend as K
from keras import models, layers, regularizers
from keras.regularizers import l1, l2
from keras.models import Sequential, model_from_json
from keras.layers import Dropout, Activation, Conv2D, MaxPooling2D,AveragePooling2D, GaussianNoise, BatchNormalization
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data_path = 'test_sets/Anina'
batch_size=50
datagen = ImageDataGenerator(validation_split=0.3)
train_batches = datagen.flow_from_directory(data_path,
                                            target_size=(203, 400),
                                           color_mode="grayscale",
                                            batch_size=batch_size,
                                            subset='training')
valid_batches = datagen.flow_from_directory(data_path,
                                            target_size=(203, 400),
                                            color_mode="grayscale",
                                            batch_size=batch_size,
                                            subset='validation')


# In[5]:


#Preparing Indermediate model
import tensorflow as tf

model = tf.keras.models.load_model('ExtractedModels/Dorsal178Model_Oct26.h5')
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer('Dense13').output)
intermediate_layer_model.summary()


# In[6]:


new_model = tf.keras.Sequential()
new_model.add(intermediate_layer_model)
# new_model.add(Dropout(0.7))
new_model.add(tf.keras.layers.Dense(2, name='Dense2'))
new_model.add(tf.keras.layers.Activation("softmax"))
#Here's the complete architecture of our model
new_model.summary()


# # Build and Train CNN

# In[7]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
# Early Stopping
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
             ModelCheckpoint(filepath='test.h5', monitor='val_loss', save_best_only=True)]


# In[ ]:


#Compile and train the model in 12 epoch
new_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = new_model.fit(train_batches, steps_per_epoch=11,
                    epochs=100,
                    callbacks=callbacks, # Early stopping
                    validation_data=valid_batches, validation_steps=5,
                    shuffle=True)


# In[8]:


#Plot acc
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='upper right')


# In[9]:


#Plot loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 1])
plt.legend(loc='lower right')


# In[ ]:





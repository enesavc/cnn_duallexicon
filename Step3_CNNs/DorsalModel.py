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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data_path = 'dorsal178'
batch_size=100
datagen = ImageDataGenerator(validation_split=0.1)
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


# # Build and Train CNN

# In[4]:


#Create the convolutional base
model = Sequential()
model.add(Conv2D(96, kernel_size=(9, 9), strides=(3, 3), name='Conv1', padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu', input_shape=(203, 400, 1)))
model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), padding="same", strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), name='Conv2',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), padding="same", strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name='Conv3',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3), padding="same", strides=(2, 2)))
model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='Conv4',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name='Conv5',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
#model.add(MaxPooling2D((3, 3), padding="same", strides=(2, 2)))
#model.add(Conv2D(2048, kernel_size=(3, 3), strides=(1, 1), name='Conv6',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
#model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
#model.add(BatchNormalization())
#model.add(MaxPooling2D((3, 3), padding="same", strides=(2, 2)))
#model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='Conv7',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
#model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
#model.add(BatchNormalization())
#model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name='Conv8',padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activation='relu'))
#model.add(GaussianNoise(0.1))
# model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(AveragePooling2D((3, 3), padding="same", strides=(2, 2)))
#Let's display the architecture of our model so far.
model.summary()


# In[5]:


#Add Dense layers on top
model.add(Flatten())
model.add(Dense(4096, activation='relu', name='Dense1'))
#model.add(Dense(2048, activation='relu', name='Dense12'))
#model.add(Dense(4096, activation='relu', name='Dense13'))
model.add(Dropout(0.1))
model.add(Dense(178, name='Dense2'))
model.add(Activation("softmax"))
#Here's the complete architecture of our model
model.summary()


# In[9]:


# #Compile and train the model in 12 epoch
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(train_batches, steps_per_epoch=(len(train_batches)*0.7)//batch_size,
#                     epochs=100,
#                     validation_data=valid_batches, validation_steps=(len(valid_batches)*0.3)//batch_size,
#                     shuffle=True)


# In[10]:


# Early Stopping
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
             ModelCheckpoint(filepath='Dorsal178Model_Oct26.h5', monitor='val_loss', save_best_only=True)]


# In[11]:


#Compile and train the model in 12 epoch
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_batches, steps_per_epoch=5000,
                    epochs=100,
                    callbacks=callbacks, # Early stopping
                    validation_data=valid_batches, validation_steps=1000,
                    shuffle=True)


# In[9]:


#Plot acc
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')


# In[10]:


#Plot loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 1])
plt.legend(loc='lower right')


# In[11]:


# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("dorsal178weights.h5")
#print("Saved model to disk")

#model.save('dorsal178model.h5')


# In[ ]:





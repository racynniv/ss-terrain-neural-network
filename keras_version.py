import sys
import os
sys.path.append("..")

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
#from cnn_utils import *
#from test_utils import summary, comparator

np.random.seed(1)

import matplotlib.pyplot as plt

tf.test.gpu_device_name()

print(tf.__version__)

from tensorflow.keras.utils import Sequence

from data_loader import load_batch

class DataGenerator(Sequence):

  def __init__(self, data, base_dir, down_scale=2, batch_size=16, noisy=False, noise_factor=0.5, shuffle=True):
  
    self.data = data
    self.data_noisy = None
    self.index = [i for i in range(self.data.shape[0])]
    self.base_dir = base_dir
    
    self.batch_size = batch_size
    self.noisy = noisy
    self.noise_factor = noise_factor
    self.shuffle = shuffle
    self.on_epoch_end()
    print(self.data.shape)
    print(int(self.data.shape[0] / self.batch_size))

  def __len__(self):
    return int(self.data.shape[0] / self.batch_size)

  def __getitem__(self, index):
    aug = np.zeros((self.batch_size,5))
    batch = np.hstack((self.data[index:index+self.batch_size],aug))
    img, accels, sds, tf, c_a = load_batch(batch,self.base_dir,down_scale=2)
    img = np.expand_dims(img, axis=-1)
    accels = np.expand_dims(accels[:,:,:,2], axis=-1)
    tf = np.expand_dims(tf,axis=-1)
    accels = np.multiply(accels,tf)
    return [img, c_a,tf], accels

  def on_epoch_end(self):
    np.random.shuffle(self.index)
    

def convolutional_model(img_shape, filters=[32,64,128,32], kernel_size=[3,3,3,3], 
                           pool_sizes=[2,2,2,2], pool_strides=[2,2,2,2]):
    """PCA's deeper brother. See instructions above. Use `code_size` in layer definitions."""
    H,W,C = img_shape
    
    img = tfl.Input(shape=img_shape)
    P = img
    
    binary_tf = tfl.Input(shape=img_shape)
    
    accel = tfl.Input(shape=(6,))
    
    denom = np.prod(pool_strides)
    
    end_size_0 = int(img_shape[0] / denom)
    end_size_1 = int(img_shape[1] / denom)
    middle_size = end_size_0 * end_size_1 * filters[-1]
    
    for i in range(len(filters)):
        Z = tfl.Conv2D(filters[i],kernel_size[i],padding='same')(P)
        A = tfl.ReLU()(Z)
        P = tfl.MaxPool2D(pool_size=(pool_sizes[i],pool_sizes[i]),
                        strides=(pool_strides[i],pool_strides[i]),padding='same')(A)
        
    F = tfl.Flatten()(P)
    
    cat = tfl.concatenate([F,accel])
    
    D = tfl.Dense(middle_size,activation='relu')(cat)
    
    Z = tfl.Reshape((end_size_0,end_size_1,filters[-1]))(D)
    
    for i in range(len(filters)-2,-1,-1):
        Z = tfl.Conv2DTranspose(filters=filters[i], kernel_size=(kernel_size[i],kernel_size[i]),
                             strides=pool_strides[i], padding='same', activation='relu')(Z)
        
    Z = tfl.Conv2DTranspose(filters=1, kernel_size=(kernel_size[0],kernel_size[0]),
                               strides=pool_strides[0],activation=None, padding='same')(Z)
    
    output = tfl.Multiply()([binary_tf,Z])
    
    model = tf.keras.Model(inputs=[img,accel,binary_tf],outputs=output)
    return model
    
if __name__=="__main__":
  stamps = np.load("/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/dirs_and_files.npy",allow_pickle=True)
  length = len(stamps[0])+len(stamps[1])+len(stamps[2])+len(stamps[3])+len(stamps[4])
  print(len(stamps[0]))
  print(len(stamps[1]))
  print(len(stamps[2]))
  print(len(stamps[3]))
  print(len(stamps[4]))
  print(length)
  stamps = stamps.flatten()
  length = len(stamps[0])
  stamps = [stamps[0]+stamps[1]+stamps[2]+stamps[3]+stamps[4]]
  print(length)
  stamps = np.squeeze(np.array(stamps))
  train = stamps[length:]
  test = stamps[:length]
  print(len(train))
  print(len(test))
  print(len(stamps))
  
  train_gen = DataGenerator(train, "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/", noisy=True)
  test_gen = DataGenerator(test, "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/", noisy=True)
  
  conv_model = convolutional_model((int(480/2), int(640/2), 1))
  conv_model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['mse'])
  conv_model.summary()
  
  model_filename = 'autoencoder.{0:03d}.hdf5'
  last_finished_epoch = None
  checkpoint_filepath = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/cp.ckpt"

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                verbose=1)
                                                                
  history = conv_model.fit_generator(generator=train_gen,
                                      validation_data=test_gen,
                                      validation_steps=2,
                                      steps_per_epoch=2,
                                      epochs=25)

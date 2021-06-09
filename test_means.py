# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from augment_image_names import Generator
from data_loader import load_batch
from cnn_model import *
import pickle as pkl
import random
from cnn_model import *
import os


files = np.concatenate((np.linspace(10,200,20,dtype=int),np.linspace(250,1000,16,dtype=int),np.linspace(1000,83000,int((83000-1000)/100 + 1),dtype=int)))

model = cnn(480,640,1,conv_featmap=[64,32,16,3],deconv_featmap=[3,16,32,64,3],
                 kernel_size=[5,5,5,5],dekernel_size=[5,5,5,5,5],conv_strides=[1,1,1,1],deconv_strides=[1,1,1,1,1],
                 pool_size=[4,2,2,2],pool_strides=[4,2,2,2],upsample_size=[2,2,2,4],fc_layer_size=[640*480*3],learning_rate=0.005,
                 lambda_l2_reg=.01,activation=tf.nn.relu)

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)

fig = plt.figure()
z = test_accels[:,:,:,2]
z = np.reshape(z,(z.shape[1],z.shape[2]))
plt.imshow(z, cmap='hot', interpolation='nearest')
fig.savefig('mean_imgs/original.png')

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))
  res = res[0]
  print(res.shape)

  fig = plt.figure()
  z_pred = np.reshape(res[:,:,2],(res.shape[0],res.shape[1]))
  plt.imshow(z_pred, cmap='hot', interpolation='nearest')
  fig.savefig('mean_imgs/prediction_{}.png'.format(i))




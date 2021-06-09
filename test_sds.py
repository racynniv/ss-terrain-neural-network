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

#files = np.linspace(15500,25400,int((25400-15500)/100 + 1),dtype=int)

model = cnn(480,640,1,conv_featmap=[64,32,16,3],deconv_featmap=[3,16,32,64,3],
                 kernel_size=[5,5,5,5],dekernel_size=[5,5,5,5,5],conv_strides=[1,1,1,1],deconv_strides=[1,1,1,1,1],
                 pool_size=[4,2,2,2],pool_strides=[4,2,2,2],upsample_size=[2,2,2,4],fc_layer_size=[640*480*3],learning_rate=0.005,
                 lambda_l2_reg=.01,activation=tf.nn.relu)
                 
tot_itr = 32

n = 1

"""

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

fig = plt.figure()
z_sd = test_sds[:,:,:,2]
z_sd = np.reshape(z_sd,(z_sd.shape[1],z_sd.shape[2]))
plt.imshow(z_sd, cmap='hot', interpolation='nearest')
fig.savefig('sd_imgs/original.png')

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))



tot_itr = 64

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))
  
  
"""
tot_itr = 100

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))
  
  
tot_itr = 32

n = 5

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))



tot_itr = 64

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))
  

  
tot_itr = 100

base_dir = "/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/"

feed_test = np.array([['terrain5/run1/', '1592968304.1', '0.0', '0.0', '0.0', '0.0', '0.0']])
test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False)
test_images = np.expand_dims(test_images, axis=-1)
noise = np.random.normal(0,n,size=(tot_itr,test_images.shape[1],test_images.shape[2],test_images.shape[3]))
test_images = np.repeat(test_images, tot_itr, axis=0) + noise
test_ca = np.repeat(test_ca, tot_itr, axis=0)

for i in files:
  i = str(i)
  res = model.predict(test_images,test_ca,pre_trained_model='/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/saved/{}'.format(i))

  z_sd = test_sds[:,:,:,2]
  pred_sd = np.std(res,axis=0)
  z_pred_sd = pred_sd[:,:,2]
  xs = z_sd.flatten()
  ys = z_pred_sd.flatten()
  zers = np.argwhere(xs==-1)
  xs[zers] = 0

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xs,ys)
  ax.set_xlabel("Uncertainty in Data")
  ax.set_ylabel("Uncertainty in Prediction")
  fig.savefig('sd_imgs/scatter_{}_{}_{}.png'.format(tot_itr,n,i))
  

  fig = plt.figure()
  z_pred_sd = np.reshape(z_pred_sd,(z_pred_sd.shape[0],z_pred_sd.shape[1]))
  plt.imshow(z_pred_sd, cmap='hot', interpolation='nearest')
  fig.savefig('sd_imgs/prediction_{}_{}_{}.png'.format(tot_itr,n,i))

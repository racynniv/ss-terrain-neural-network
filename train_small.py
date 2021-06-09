# Import modules
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import numpy as np
from matplotlib import pyplot as plt
import cv2
from augment_image_names import Generator
from data_loader import load_batch
from cnn_model import *

#load data
import os
stamps = np.load("/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/dirs_and_files.npy",allow_pickle=True)
length = len(stamps[0])+len(stamps[1])+len(stamps[2])+len(stamps[3])+len(stamps[4])
stamps = stamps.flatten()
#length = len(stamps[0])
length_n = 2
#print(stamps[1])
stamps = [stamps[0]+stamps[1]+stamps[2]+stamps[3]+stamps[4]]
indices = np.linspace(0,len(stamps[0])-1,length_n,dtype=int)
print(indices)
stamps = np.squeeze(np.array(stamps))
train = stamps[indices]
test = stamps[indices]
print(train.shape)
print(test.shape)

gen = Generator(train,translate=[5,5],flip=[0,0],noise=1)

print(gen.aug_size())
next_batch = gen.gen_batch(1)

batch = next(next_batch)

print(batch.shape)

from data_loader import load_batch

images, arrays, sds, dicts, accels = load_batch(batch,"/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/")
print(len(images))
print(accels.shape)
print(dicts.shape)

print(tf.config.list_physical_devices('GPU'))

if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")
   
from cnn_model import *

scale = 2

model = cnn(int(480/scale),int(640/scale),1,conv_featmap=[32,64,128,32],deconv_featmap=[32,128,64,32,1],
                 kernel_size=[3,3,3,3],dekernel_size=[3,3,3,3,3],conv_strides=[1,1,1,1],deconv_strides=[2,2,2,2,2],
                 pool_size=[1],pool_strides=[1],upsample_size=[1,1],fc_layer_size=[640*480*3],learning_rate=0.001,
                 lambda_l2_reg=.01,activation=tf.nn.relu)
                 
model.train(train,test,"/media/ros/de64b3cc-d3b4-44e1-b807-300d3d8adb21/ss_terrain_nav/data/",epochs=5000000,
              batch_size=1,translate=[0,0],
              flip=[1,0],noise=0,scale_img=scale,model_name=None)

model.plot()

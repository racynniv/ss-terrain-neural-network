import numpy as np
import cv2
import glob, os
import pickle as pkl
import random
from skimage.measure import block_reduce

def load_batch(batch,base_dir,train=True,size=[480,640],down_scale=1):

  width = int(size[1] * 1/down_scale)
  height = int(size[0] * 1/down_scale)

  images = np.empty((batch.shape[0],height,width))
  accels = np.zeros((batch.shape[0],height,width,3))
  sds = np.ones((batch.shape[0],height,width,3))
  tf = np.zeros((batch.shape[0],height,width))
  curr_a = np.zeros((batch.shape[0],6))
  os.chdir(base_dir)
  augs = np.array(batch[:,2:],dtype=float)
  for i in range(batch.shape[0]):
    r = batch[i]
    a = augs[i]
    time = round(float(r[1]),2)
    image = (cv2.imread(r[0] + "depth_{}.png".format(time),0)/255)*2-1
    sd_and_v = np.load(r[0] + "{}.npy".format(time),0)
    ca = np.load(r[0] + "accel_{}.npy".format(time),0)
    curr_a[i] = ca.flatten()
    value = sd_and_v[:,:,:3]
    sd = sd_and_v[:,:,3:]
    
    dim = (width, height)
    
    image = cv2.resize(image,dim)
    value = block_reduce(value,block_size=(down_scale,down_scale,1),func=np.mean)
    sd = block_reduce(sd,block_size=(down_scale,down_scale,1),func=np.mean)
    
    
    """
    with open(r[0] + '{}.pkl'.format(time), 'rb') as f:
      u = pkl._Unpickler(f)
      u.encoding = 'latin1'
      p = u.load()
      
    value,sd = dict_to_accel(p,value)
    """
    
    if train:
      if a[0] != 0 or a[1] != 0:
        images[i] = np.roll(image,shift=(int(a[0]),int(a[1])),axis=(0,1))
        accels[i] = np.roll(value,shift=(int(a[0]),int(a[1])),axis=(0,1))
        sds[i] = np.roll(sd,shift=(int(a[0]),int(a[1])),axis=(0,1))
        
      if a[2] != 0 or a[3] != 0:
        if a[2] != 0 and a[3] !=0:
          flip_i = np.flip(image,axis=0)
          value_i = np.flip(value,axis=0) * -1
          sd_i = np.flip(sd,axis=0)
          images[i] = np.flip(flip_i,axis=1)
          accels[i] = np.flip(value_i,axis=1)
          sds[i] = np.flip(sd_i,axis=1)
        elif a[2] != 0:
          images[i] = np.flip(image,axis=0)
          accels[i] = np.flip(value,axis=0) * -1
          sds[i] = np.flip(sd,axis=0)
        else:
          images[i] = np.flip(image,axis=1)
          accels[i] = np.flip(value,axis=1)
          sds[i] = np.flip(sd,axis=1)
          
      elif a[4] != 0:
        images[i] = image+np.random.normal(0,a[4],image.shape)
        accels[i] = value
        sds[i] = sd
    
      else:
        images[i] = image
        accels[i] = value
        sds[i] = sd
        
    else:
      images[i] = image
      accels[i] = value
      sds[i] = sd
    if a[2] == 0:
      tf[i] = accels[i,:,:,0] != -1
    else:
      tf[i] = accels[i,:,:,0] != 1
  
  return images, accels, sds, tf, curr_a
  
def dict_to_accel(dic, values):
  accels = np.zeros((values.shape[0],values.shape[1],3))
  sds = np.ones((values.shape[0],values.shape[1],3))
  indices = np.argwhere(values[:,:,0]>-1)
  for i in indices:
    tup_a = tuple((i[0],i[1],0))
    tup_sd = tuple((i[0],i[1],1))
    accels[i[0],i[1]] = list(dic[tup_a])[0]
    sds[i[0],i[1]] = list(dic[tup_sd])
    """
    #print(ac)
    accels[i[0],i[1]] = random.choice(ac)
    if len(ac) != 1:
        ac = np.array(ac)
        sds[i[0],i[1],0] = np.std(ac[:,0])
        sds[i[0],i[1],1] = np.std(ac[:,1])
        sds[i[0],i[1],2] = np.std(ac[:,2])
    else:
        sds[i[0],i[1],0] = sds[i[0],i[1],1] = sds[i[0],i[1],2] = 1
    """
  return accels, sds

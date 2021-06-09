import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import time
from augment_image_names import Generator
from data_loader import load_batch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math

"""

https://github.com/richzhang/colorization/blob/master/colorization/models/colorization_deploy_v2.prototxt

This object is a convolutional neural network using tensorflow and a generator
with data autmentation. The object initializes with scalars height, width, 
channels, output size, learning rate, and l2 regularization. It also has a list 
of ints as the units for the convoluational layers, fully connected layers, 
kernel size, pool and convolutional stride lengths. There is also a list of 
floats for the keep probability values for each fully connected layer. Finally, 
there is a variable for the fully connected activation function and the type of 
output (classification vs regression). All of the lists pertaining to the 
convolutional (and pooling) layers must be of the same length. The same holds 
true for the fully connected layers. Finally, there is the activation function, 
which stays consistent across all fully connected layers (except for the output 
layer). The network takes batches of image data, puts it through the 
convolutional layers (each followed by a pooling layer) with parameters denoted 
by the conv_featmap, kernel_size, conv_strides, pool_size, and pool_strides 
lists. This output is then funneled to the fully connected layers with parameter
lists fc_layer_size and train_keep_prob. The output of these layers is 
determined by the output size int. The training function of the object takes 
batches of any size but must be an NxM matrix input. Using the train function, 
users can train on presplit data and can alter the parameters of batch size, 
number of epochs, translation, flips, rotations, and added noise. This trained 
network is automatically saved under the given name and can be loaded for 
further training. There is also a predict function that loads an input pre 
trained model or loads the most recent checkpoint for predictions.
"""

class cnn(object):
    def __init__(self,height,width,channels,conv_featmap=[32,64],deconv_featmap=[64,32,3],
                 kernel_size=[5,5],dekernel_size=[5,5,5],conv_strides=[1,1],deconv_strides=[1,1,1],
                 pool_size=[2,2],pool_strides=[2,2],upsample_size=[2,2],fc_layer_size=[1024],
                 learning_rate=0.01,lambda_l2_reg=.01,activation=tf.nn.relu):
        # Ensures that the hidden layers have corresponding keep probs
        assert len(conv_featmap) == len(kernel_size) and len(conv_featmap) == len(pool_size)


        # Sets variables for later use
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_featmap = conv_featmap
        self.deconv_featmap = deconv_featmap
        self.kernel_size = kernel_size
        self.dekernel_size = dekernel_size
        self.conv_strides = conv_strides
        self.deconv_strides = deconv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.upsample_size = upsample_size
        self.fc_layer_size = fc_layer_size
        self.learning_rate = learning_rate
        self.lambda_l2_reg = lambda_l2_reg
        self.activation = activation

        # Creates NN using private functions
        self.__input_layer()
        #self.__bn_input()
        self.__conv_layers()
        self.__middle_layer()
        #self.__deconv_layers()
        #self.__fc_layers()
        self.__output_layer()
        self.__loss()
        self.__optimizer()
        self.saver = tf.train.Saver(max_to_keep=0)

    # Creates the input layer using placeholders (assumes batches x 3 dim input)
    def __input_layer(self):
        self.inputs = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels))
        self.targets = tf.placeholder(tf.float32,shape=(None,self.height,self.width,1))
        self.sds = tf.placeholder(tf.float32,shape=(None,self.height,self.width,3))
        self.output_tf = tf.placeholder(tf.float32,shape=(None,self.height,self.width,1))
        self.f_div = tf.placeholder(tf.float32,shape=(self.height,self.width,1))
        self.curr_a = tf.placeholder(tf.float32,shape=(None,6))
        self.is_train = tf.placeholder(tf.bool)
        
    def __bn_input(self):
        self.bn_input = tf.layers.batch_normalization(self.inputs,training=self.is_train,axis=[1,2,3])
        
    # Creates convolutional layers (each conv layer followed by a pooling layer)
    def __conv_layers(self):
        self.i_conv = tf.layers.conv2d(self.inputs, filters=self.conv_featmap[0],
                                     kernel_size=self.kernel_size[0],strides=self.conv_strides[0],
                                     padding="SAME",activation=self.activation)
        #self.bn = tf.layers.batch_normalization(self.i_conv, training=self.is_train,axis=[1,2,3])
        self.c_pool = tf.nn.max_pool(self.i_conv, ksize=[1,self.pool_size[0],self.pool_size[0],1],
                                   strides=[1,self.pool_strides[0],self.pool_strides[0],1],padding="VALID")
        print(self.c_pool.shape)

        for i in range(1,len(self.conv_featmap)):
            self.conv = tf.layers.conv2d(self.c_pool, filters=self.conv_featmap[i],kernel_size=self.kernel_size[i],
strides=self.conv_strides[i],padding="SAME",activation=self.activation)
            print(self.conv.shape)
            #self.bn = tf.layers.batch_normalization(self.conv, training=self.is_train,axis=[1,2,3])
            self.c_pool = tf.nn.max_pool(self.conv, ksize=[1,self.pool_size[i],self.pool_size[i],1],
                                       strides=[1,self.pool_strides[i],self.pool_strides[i],1],padding="VALID")
            print(self.c_pool.shape)
            
    def __middle_layer(self):
        shape = self.c_pool.get_shape()
        print(shape)
        size = shape[1]*shape[2]*shape[3]
        self.middle = tf.reshape(self.c_pool,shape=[-1,size])
        self.middle = tf.concat([self.middle,self.curr_a],1)
        #self.bn = tf.layers.batch_normalization(self.middle, training=self.is_train)
        print(self.middle.shape)
        self.middle = tf.layers.dense(self.middle,size)
        print(1)
        print(self.middle.shape)
        self.middle = tf.reshape(self.middle,shape=[-1,shape[1],shape[2],shape[3]])
        print(self.middle.shape)
    
    def __deconv_layers(self):
      self.d_pool = self.middle
      for i in range(len(self.deconv_featmap)-1):
        self.deconv = tf.layers.conv2d(self.d_pool, filters=self.deconv_featmap[i],kernel_size=self.dekernel_size[i],
                                              strides=self.deconv_strides[i],padding="SAME",activation=self.activation)
        print(self.deconv.shape)
        print(1)
        #self.bn = tf.layers.batch_normalization(self.deconv, training=self.is_train,axis=[1,2,3])
        print(self.deconv.shape)
        new_h = tf.shape(self.deconv)[1]*self.upsample_size[i]
        new_w = tf.shape(self.deconv)[2]*self.upsample_size[i]
        self.d_pool = tf.image.resize(self.deconv, size=[new_h,new_w])
        print(self.d_pool.shape)
        
      self.deconv_f = tf.layers.conv2d(self.d_pool, filters=self.deconv_featmap[-1],kernel_size=self.dekernel_size[-1],
                                     strides=self.deconv_strides[-1],padding="SAME",activation=None)
                                     
      print(self.deconv_f.shape)
                                     
    """
    def __fc_layers(self):
        shape = self.deconv_f.get_shape()
        print(shape[1].value)
        size = 640*480*3
        self.fc = tf.reshape(self.deconv_f,shape=[-1,size])
        self.fc = tf.layers.dense(self.fc,size)
    """

    # Takes output of FC layer and creates an output of output_size
    # (This does not have dropout because it is the output layer)
    def __output_layer(self):
        self.output = tf.reshape(self.middle,[-1,self.height,self.width,1])
        #self.output = tf.multiply(self.output_tf,self.mult)

    # Defines loss based on if the output is a regression or classification. If 
    # classification, use softmax, if regression, use mean squared error
    def __loss(self):
        self.loss = (self.targets - self.output)**2
        #self.loss = tf.losses.mean_squared_error(self.targets,self.output)
        """
        self.error = self.targets-self.output
        self.mse = (self.targets-self.output)**2
        self.sum_mse = tf.math.reduce_sum(self.mse,axis=0)
        self.sum_tf = tf.math.reduce_sum(self.output_tf,axis=0)
        self.loss = tf.math.divide(self.sum_mse,self.sum_tf+1e-8)
        """
        """
        zscore = tf.math.divide(self.targets-self.output,self.sds)
        loss_sum = tf.math.reduce_sum(zscore**2,axis=0)
        self.loss = tf.math.divide(loss_sum,self.f_div)
        """
        
        #div_sd = np.divide(mse,self.sds)
        #self.loss = tf.reduce_sum(div_sd)/n

    # Minimizes the loss using an Adam optimizer
    def __optimizer(self):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
      """
      opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      grad, varis = zip(*opt.compute_gradients(self.loss))
      grad, _ = tf.clip_by_global_norm(grad,1)
      self.optimizer = opt.apply_gradients(zip(grad,varis))
      """

    # Trains the network based on the train inputs given and uses the test set
    # to test accuracy on a non training set
    def train(self,train_names,test_names,base_dir,epochs=20,
              batch_size=64,test_batch_size=1,tot_std_itr=50,translate=[0,0],
              flip=[0,0],noise=0,std_noise=5,scale_img=1,model_name=None,pre_trained_model=None):

        # Create the generator to output batches of data with given transforms
        gen = Generator(train_names,translate=translate,flip=flip,noise=noise)
        next_batch = gen.gen_batch(batch_size)
        
        test_gen = Generator(test_names)
        test_batch = test_gen.gen_batch(test_batch_size)

        # Set number of iterations (SIZE CAN BE CHANGED BECAUSE OF GENERATOR)
        aug_size = gen.aug_size()
        iters = int(aug_size / batch_size)
        print('number of batches for training: {}'.format(iters))
        
        # Set base levels and model name
        iter_tot = 0
        best_mse = 100000000
        best_sd = 100000000
        old_p = np.zeros((test_batch_size,480,640,3))
        self.losses = []
        if model_name == None:
            cur_model_name = 'basic_model'

        # Start session, initialize variables, and load pretrained model if any
        with tf.Session() as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format('model'),
                                           sess.graph)
            sess.run(tf.global_variables_initializer())
            old_mse = 0
            old_sd = 0
            if pre_trained_model != None:
                try:
                    print("Loading model from: {}".format(pre_trained_model))
                    self.saver.restore(sess,'{}'.format(pre_trained_model))
                except Exception:
                    raise ValueError("Failed Loading Model")
                    
            start_bin = True

            # Set up loops for epochs and iterations per epochs
            for epoch in range(epochs):
                #print("epoch {}".format(epoch + 1))

                for itr in range(iters):
                    merge = tf.summary.merge_all()
                    
                    iter_tot += 1
                    #print(iter_tot)

                    # Create feed values using the generator
                    feed_names = next(next_batch)
                    feed_image, feed_accels, feed_sd, feed_tf, feed_ca = load_batch(feed_names,base_dir,down_scale=scale_img)
                    #print(len(np.argwhere(feed_sd==0)))
                    feed_image = np.expand_dims(feed_image, axis=-1)
                    #print(np.max(feed_image))
                    feed_f_div = np.sum(feed_tf,axis=0)
                    zer = np.argwhere(feed_f_div == 0)
                    feed_f_div[zer[:,0],zer[:,1]] = 1
                    feed_f_div = np.reshape(feed_f_div,(feed_f_div.shape[0],feed_f_div.shape[1],1))
                    feed_accels = np.round(feed_accels[:,:,:,2]*100,decimals=0)
                    feed_accels = np.reshape(feed_accels,(feed_accels.shape[0],feed_accels.shape[1],feed_accels.shape[2],1))
                    
                    #feed_tf = np.stack((feed_tf,feed_tf,feed_tf),axis=-1)
                    feed_tf = np.reshape(feed_tf,(feed_tf.shape[0],feed_tf.shape[1],feed_tf.shape[2],1))
                    feed_accels = np.multiply(feed_accels,feed_tf)
                    feed = {self.inputs: feed_image, self.targets: feed_accels, self.sds: feed_sd, self.output_tf: feed_tf, self.curr_a: feed_ca, self.f_div: feed_f_div, self.is_train: True}

                    # Feed values to optimizer and output loss (for printing)
                    _, cur_loss, output= sess.run([self.optimizer,self.loss,self.output],
                                           feed_dict=feed)
                                           
                    if start_bin:
                        duration = 1  # seconds
                        freq = 440  # Hz
                        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
                        os.system('spd-say "Training Started"')
                        start_bin = False
                        
                    #print(np.max(cur_loss))
                    #cl = np.sum(cur_loss)/np.count_nonzero(output)
                    #a = np.count_nonzero(output)
                    """
                    print(np.max(init))
                    print(np.max(conv))
                    print(np.max(middle))
                    print(np.max(deconv))
                    print(np.max(cur_loss))
                    print(cur_loss)
                    print(np.max(np.array(cur_loss)))
                    print(np.argmax(cur_loss))
                    cur_loss = np.array(cur_loss)
                    print(cur_loss.shape)
                    where = np.unravel_index(cur_loss.argmax(),cur_loss.shape)
                    print(np.unravel_index(cur_loss.argmax(),cur_loss.shape))
                    """
                    #print(cur_loss.shape)
                    #print(feed_tf.shape)
                    #print(feed_tf[0,np.unravel_index(cur_loss.argmax(),cur_loss.shape)])
                    #print(error[0,where[0],where[1]])
                    #print(output[0,where[0],where[1]])
                    if iter_tot%5 == 0:
                      print(iter_tot)
                      print(np.max(feed_image))
                      print(np.min(feed_image))
                      print("loss")
                      print(np.max(cur_loss))
                      print("outputs")
                      print(np.max(output))
                      print(np.min(output))
                      print("labels")
                      print(np.max(feed_accels))
                      print(np.min(feed_accels))
                      """
                      print("mse sum")
                      print(np.sqrt(np.max(sum_mse)))
                      print(np.sqrt(np.min(sum_mse)))
                      print("tf sum")
                      print(np.max(sum_tf))
                      print(np.min(sum_tf))
                      
                      print("accel")
                      print(feed_ca)
                      print("error")
                      print(np.max(error))
                      print(np.min(error))
                      """
                      if math.isnan(np.max(cur_loss)) or math.isnan(np.max(output)):
                        os.system('spd-say "Not a Number"')
                        return
                      elif np.max(output) > 1e9:
                        os.system('spd-say "Too Large"')
                        return
                      """
                      print("errors")
                      print(np.max(mse))
                      print(np.min(mse))
                      arg_mx = np.unravel_index(np.argmax(mse),mse.shape)
                      arg_mn = np.unravel_index(np.argmin(mse),mse.shape)
                      print("diff locations max")
                      print(output[arg_mx])
                      print(feed_accels[arg_mx])
                      print("diff locations min")
                      print(output[arg_mn])
                      print(feed_accels[arg_mn])
                      print("targets")
                      print(np.max(abs(output-feed_accels)))
                      print(np.min(abs(output-feed_accels)))
                    #print(np.max(feed_accels))
                    """
                    """
                    print(cl)
                    print(np.count_nonzero(feed_tf))
                    print(a)
                    print(np.max(cur_loss))
                    print(np.max(output))
                    print(np.max(feed_accels))
                    """
                    #print('here')
                    
                    diff = output-feed_accels
                    mean = np.mean((diff)**2)
                    std = np.std((diff))
                    #print(mean)
                    #print(std)
                    #print(iter_tot)
                    self.losses.append(np.mean(cur_loss))

                    # After 100 iterations, check if test accuracy has increased
                    if iter_tot % 100 == 0:
                        #print(iter_tot)
                        feed_test = next(test_batch)
                        test_images, test_accels, test_sds, test_tf, test_ca = load_batch(feed_test,base_dir,train=False,down_scale=scale_img)
                        test_images = np.expand_dims(test_images, axis=-1)
                        #test_tf = np.stack((test_tf, test_tf, test_tf),axis=-1)
                        test_accels = np.round(test_accels[:,:,:,2]*100,decimals=0)
                        test_accels = np.reshape(test_accels,(test_accels.shape[0],test_accels.shape[1],test_accels.shape[2],1))
                        test_tf = np.reshape(test_tf,(test_tf.shape[0],test_tf.shape[1],test_tf.shape[2],1))
                        test_accels = np.multiply(test_accels,test_tf)
                        output = sess.run([self.output],feed_dict={self.inputs:
                                        test_images, self.targets: test_accels, self.sds: test_sds, 
                                        self.output_tf: test_tf, self.curr_a: test_ca, 
                                        self.is_train: False})
                        diff = (np.multiply(output,test_tf)-test_accels)**2
                        mse = np.sum(diff)/np.sum(test_tf)
                        print("MSE")
                        print(best_mse)
                        print(mse)
                        if mse < best_mse-.1:
                            print('Best validation accuracy! iteration:'
                                  '{} mse: {}%'.format(iter_tot, mse))
                            best_mse = mse
                            self.saver.save(sess,'./saved/mse_{}_{}'.format(iter_tot,mse))
                            print("Saved")
                            if best_mse < .2:
                              os.system('spd-say "Cutoff"')
                              return
                      
                        """
                        preds=np.empty((tot_std_itr,test_images.shape[0],test_images.shape[1],test_images.shape[2],test_images.shape[3]))
                        for std_itr in range(tot_std_itr):
                            test_noise = np.random.randint(-1*std_noise,std_noise+1,(test_images.shape[0],test_images.shape[1],test_images.shape[2],test_images.shape[3]))
                            pred = sess.run([self.output],feed_dict={self.inputs:
                                        test_images+test_noise, self.targets: test_accels, self.sds: test_sds, 
                                        self.output_tf: test_tf, self.curr_a: test_ca, 
                                        self.is_train: False})
                            preds[std_itr] = pred
                        std = np.std(preds,axis=0)
                        std = std/np.max(std)
                        test_sds = test_sds/np.max(test_sds)
                        #print(std.shape)
                        #print(test_sds.shape)
                        s1 = np.sum((std-test_sds)**2,axis=0)
                        s2 = np.sum(s1)
                        sd = s2/np.count_nonzero(test_tf)
                        old_sd = sd
                        
                        if sd < best_sd:
                            print('Best validation accuracy! iteration:'
                                  '{} sd: {}%'.format(iter_tot, sd))
                            best_sd = sd
                            self.saver.save(sess,'./saved/sd_{}_{}'.format(iter_tot,sd))
                            print("Saved")
                        
                        """
                        
                    """
                        if iter_tot < 200:
                            self.saver.save(sess,'./saved/{}'.format(iter_tot))
                        elif iter_tot % 50 == 0 and iter_tot < 1000:
                            self.saver.save(sess,'./saved/{}'.format(iter_tot))
                        elif iter_tot % 100 == 0:
                            self.saver.save(sess,'./saved/{}'.format(iter_tot))
                            """
                    if iter_tot % 10000 == 0:
                      self.saver.save(sess,'./saved/{}'.format(iter_tot))

        print("Traning ends. The best valid accuracy is {}." \
               " Model named {}.".format(best_mse, cur_model_name))

    # Plot training losses from most recent session
    def plot(self):
        plt.plot(self.losses)
        plt.ylim(0,1000)

    # Predicts class of input based on pre trained model
    def predict(self,x,curr_a,pre_trained_model=None):
        print(self.inputs.get_shape()[1:])
        assert(x.shape[1:] == self.inputs.get_shape()[1:])

        self.session = tf.Session()
        with self.session as sess:
            saver = tf.train.Saver()
            if pre_trained_model != None:
                try:
                    print("Loading model from: {}".format(pre_trained_model))
                    #saver = tf.train.import_meta_graph('{}.meta'.format(pre_trained_model))
                    saver.restore(sess,pre_trained_model)
                except Exception:
                    raise ValueError("Failed Loading Model")
            binary_tf = np.ones((x.shape[0],x.shape[1],x.shape[2],1))
            #rand = np.random.randint(-1*noise,noise+1,(x.shape[0],x.shape[1],x.shape[2],1))
            pred = sess.run([self.output],feed_dict={self.inputs: x, self.output_tf: binary_tf, self.curr_a: curr_a, self.is_train: False})
            #rpred = sess.run([self.output],feed_dict={self.inputs: rand, self.output_tf: binary_tf, self.curr_a: curr_a})
            
            return pred[0]
            
            """
            for i in range(num):
                rand = np.random.randint(-1*noise,noise+1,(x.shape[0],x.shape[1],x.shape[2],1))
                img = x + rand
                pred = sess.run([self.output],feed_dict={self.inputs: img, self.output_tf: binary_tf, self.curr_a: curr_a})
                print(pred[0].shape)
                preds[i] = pred[0]
            return preds
            """
            


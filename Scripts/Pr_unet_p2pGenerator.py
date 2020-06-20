# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:44:34 2018

@author: Moha-Thinkpad
"""

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import datetime
import numpy as np
#import matplotlib 
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow.keras

import argparse
import tensorflow as tf


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))
import glob
import os
from skimage import io
import gc


####################################
########################################################################
####################################


from tensorflow.keras.callbacks import Callback
class callback_4_StopByLossValue(Callback):
    
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:            
            print("Epoch %05d: reached desired error at epoch" % epoch)
            self.model.stop_training = True


def custom_loss (y_true, y_pred):
    
    
    #A = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
    B = tensorflow.keras.losses.mean_absolute_error(y_true, y_pred)
    
    return(B)


sum_dim_channel = Lambda(lambda xin: K.sum(xin, axis=3))
def lrelu(x): #from pix2pix code
    a=0.2
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def lrelu_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)
layer_lrelu=Lambda(lrelu, output_shape=lrelu_output_shape)


def PreProcess(InputImages):
    
    #output=np.zeros(InputImages.shape,dtype=np.float)
    InputImages=InputImages.astype(np.float)
    for i in range(InputImages.shape[0]):
        try:
            InputImages[i,:,:,:]=InputImages[i,:,:,:]/np.max(InputImages[i,:,:,:])
#            output[i,:,:,:] = (output[i,:,:,:]* 2)-1
        except:
            InputImages[i,:,:]=InputImages[i,:,:]/np.max(InputImages[i,:,:])
#            output[i,:,:] = (output[i,:,:]* 2) -1
            
    return InputImages


####################################
########################################################################
####################################

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test", "export"])
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint",  help="where to ")
parser.add_argument("--input_dir_all", help="path to folder containing images")
parser.add_argument("--output_dir_all", help="where to put output files")
parser.add_argument("--cv_info_dir", default=None, help="directory contains cross validation set ups")
parser.add_argument("--task_No", default=None, help="number of task, 1 means t1, 2 means t2, 3 means multi")
parser.add_argument("--desired_l1_loss", default=0.005, help="stop parameter: desired l1 loss for early stop the training < max_epochs ")
parser.add_argument("--lr",  help="adam learning rate")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--kernelsize", type=int, default=4, help="kernelsize for the conv layers of generator")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")


# export options
a = parser.parse_args()

#a.scale_size=512
#CROP_SIZE = 512
#
#
#a.batch_size=25
#a.max_epochs=500 #60
#a.lr=0.0002
#a.beta1=0.5
#a.ngf=100
#a.kernelsize=3
#a.seed=35555
#a.task_No='1'
#a.desired_l1_loss=0.01
#
#a.input_dir_all='./4_save/T1_to_FLAIR_T1Inv_512/ImageData'
#a.output_dir_all='./4_save/T1_to_FLAIR_T1Inv_512/Outputs_unet_p2p'
#a.cv_info_dir='./4_save/T1_to_FLAIR_T1Inv_512/CV_Info_LSliceOut'

print(a)

def CreateModel():
    
    ########### network    
    kernelSize=(a.kernelsize,a.kernelsize)
    InputLayer=tensorflow.keras.layers.Input(shape=(a.scale_size,a.scale_size,3))
    e_1=tensorflow.keras.layers.Conv2D(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(InputLayer)
    
    e_2=layer_lrelu(e_1)
    e_2=tensorflow.keras.layers.Conv2D(a.ngf * 2, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_2)
    e_2=tensorflow.keras.layers.BatchNormalization()(e_2)
    
    e_3=layer_lrelu(e_2)
    e_3=tensorflow.keras.layers.Conv2D(a.ngf * 4, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_3)
    e_3=tensorflow.keras.layers.BatchNormalization()(e_3)

    e_4=layer_lrelu(e_3)
    e_4=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_4)
    e_4=tensorflow.keras.layers.BatchNormalization()(e_4)

    e_5=layer_lrelu(e_4)
    e_5=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_5)
    e_5=tensorflow.keras.layers.BatchNormalization()(e_5)

    e_6=layer_lrelu(e_5)
    e_6=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_6)
    e_6=tensorflow.keras.layers.BatchNormalization()(e_6)

    e_7=layer_lrelu(e_6)
    e_7=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_7)
    e_7=tensorflow.keras.layers.BatchNormalization()(e_7)

    e_8=layer_lrelu(e_7)
    e_8=tensorflow.keras.layers.Conv2D(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(e_8)
    e_8=tensorflow.keras.layers.BatchNormalization()(e_8)

    
    
    
    d_8=e_8
    d_8=tensorflow.keras.layers.Activation('relu')(d_8)
    d_8=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_8)
    d_8=tensorflow.keras.layers.BatchNormalization()(d_8)
    d_8=tensorflow.keras.layers.Dropout(0.5)(d_8)
    
    d_7=tensorflow.keras.layers.concatenate(inputs=[d_8, e_7], axis=3)
    d_7=tensorflow.keras.layers.Activation('relu')(d_7)
    d_7=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_7)
    d_7=tensorflow.keras.layers.BatchNormalization()(d_7)
    d_7=tensorflow.keras.layers.Dropout(0.5)(d_7)  
    
    d_6=tensorflow.keras.layers.concatenate(inputs=[d_7, e_6], axis=3)
    d_6=tensorflow.keras.layers.Activation('relu')(d_6)
    d_6=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_6)
    d_6=tensorflow.keras.layers.BatchNormalization()(d_6)
    d_6=tensorflow.keras.layers.Dropout(0.5) (d_6)
    
    d_5=tensorflow.keras.layers.concatenate(inputs=[d_6, e_5], axis=3)
    d_5=tensorflow.keras.layers.Activation('relu')(d_5)
    d_5=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 8, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_5)
    d_5=tensorflow.keras.layers.BatchNormalization()(d_5)
    d_5=tensorflow.keras.layers.Dropout(0.5) (d_5)
    
    d_4=tensorflow.keras.layers.concatenate(inputs=[d_5, e_4], axis=3)
    d_4=tensorflow.keras.layers.Activation('relu')(d_4)
    d_4=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 4, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_4)
    d_4=tensorflow.keras.layers.BatchNormalization()(d_4)
    
    d_3=tensorflow.keras.layers.concatenate(inputs=[d_4, e_3], axis=3)
    d_3=tensorflow.keras.layers.Activation('relu')(d_3)
    d_3=tensorflow.keras.layers.Conv2DTranspose(a.ngf * 2, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_3)
    d_3=tensorflow.keras.layers.BatchNormalization()(d_3)
    
    d_2=tensorflow.keras.layers.concatenate(inputs=[d_3, e_2], axis=3)
    d_2=tensorflow.keras.layers.Activation('relu')(d_2)
#        d_2=tensorflow.keras.layers.Conv2DTranspose(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_2)
    d_2=tensorflow.keras.layers.Conv2DTranspose(a.ngf, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_2)
    d_2=tensorflow.keras.layers.BatchNormalization()(d_2)
    
    
    d_1=tensorflow.keras.layers.concatenate(inputs=[d_2, e_1], axis=3)
    d_1=tensorflow.keras.layers.Activation('relu')(d_1)
    d_1=tensorflow.keras.layers.Conv2DTranspose(3, kernel_size=kernelSize, strides=(2, 2), dilation_rate=(1, 1), padding='same',)(d_1)        
    OUT=tensorflow.keras.layers.Activation('tanh', name='last_layer_of_decoder')(d_1)
            
    model_unet=Model(inputs=InputLayer, outputs=OUT)
 
    # model_unet.summary()
    return model_unet
    ###########Train
        


import random    
import pickle
from tensorflow.keras.models import load_model

def main():
       
    
    if a.mode=='test':
        
        checkpoint_model_file=a.checkpoint+'/Model'
        
        print('loading model ...')
        MODEL_unet=load_model(checkpoint_model_file+'_weights.h5', custom_objects={ 
                                                                                    'custom_loss': custom_loss, 
                                                                                    'layer_lrelu':layer_lrelu, 
                                                                                    'lrelu':lrelu, 
                                                                                    'lrelu_output_shape':lrelu_output_shape,
                                                                                    'tf': tf}) 
    
        print('model is loaded ')

        Y_pred=MODEL_unet.predict(X_test)  
        
        for i in range(len(list_test)):
            
            filename_=list_test[i]
            try:
                filename_=filename_.replace('\n','')
            except:
                no=0
                        
            io.imsave(a.output_dir+'/'+filename_[:-4]+'-outputs.png', 255*Y_pred[i,:,:,:])
            io.imsave(a.output_dir+'/'+filename_[:-4]+'-inputs.png', 255*X_test[i,:,:,:])
            io.imsave(a.output_dir+'/'+filename_[:-4]+'-targets.png', 255*Y_test[i,:,:,:])
        
      
            
    
    if a.mode=='train':
        
        
    #    plt.figure()
    #    plt.imshow(X_train[90,:,:,:])
    #    plt.figure()
    #    plt.imshow(Y_train_heatmap[90,:,:,4])
        
 
        print('======== new training ...')
        checkpoint_model_file=a.checkpoint+'/Model'               
        MODEL_unet=CreateModel()
        print('trainable_count =',int(np.sum([K.count_params(p) for p in set(MODEL_unet.trainable_weights)])))
        print('non_trainable_count =', int(np.sum([K.count_params(p) for p in set(MODEL_unet.non_trainable_weights)])))   
        MODEL_unet.summary()
            
        # fix random seed for reproducibility
        seed = a.seed    
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        
        #### compile and train the model
        MyCallbacks = callback_4_StopByLossValue(monitor='loss', value=a.desired_l1_loss, verbose=1)    

        UsedOptimizer=optimizers.Adam(lr=a.lr, beta_1=a.beta1)
        MODEL_unet.compile(loss=custom_loss, optimizer=UsedOptimizer)        
        History=MODEL_unet.fit(X_train, Y_train,
                batch_size=a.batch_size, shuffle=True, validation_split=0.02,
            epochs=a.max_epochs,
                verbose=1, callbacks=[MyCallbacks])
        
        
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.grid()
        plt.savefig(a.output_dir+'/History_'+str(a.lr)+'.png')
        plt.close()
        
                
        Dict={'History_loss_train':History.history['loss'],
              'History_loss_val':History.history['val_loss'],}
        pickle.dump( Dict, open(a.output_dir+'/History_'+str(a.lr)+'.pkl', "wb" ) )
        
       
        print('===========training done=================')
        print('============================')
        print(datetime.datetime.now())
        print('============================')
        print('============================')
                    
            
        print('Saving model ...')
        MODEL_unet.save(checkpoint_model_file+'_weights.h5')
    
    
    


CvDirs = glob.glob(os.path.join(a.cv_info_dir, "*"))
SetNames = [os.path.basename(x) for x in CvDirs]

for cv in range(0,len(CvDirs)): # #################

    
    print('#####################')
    print('### cv:',SetNames[cv],'########')
    print('#####################')
    trainfile=a.cv_info_dir+'/'+SetNames[cv]+'/train.txt'
    testfile=a.cv_info_dir+'/'+SetNames[cv]+'/test.txt'
    
    text_file = open(trainfile)
    list_train = text_file.readlines()
    text_file.close()
    text_file = open(testfile)
    list_test = text_file.readlines()
    text_file.close()
        
    a.mode="train"
    print('############',a.mode)
    tf.reset_default_graph()
    
    a.output_dir=a.output_dir_all+'/unet_ST/Models_t'+a.task_No+'/'+SetNames[cv]    
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    
    Images_input=np.zeros((len(list_train),a.scale_size,a.scale_size,3),dtype=np.uint8)    
    Images_target=np.zeros((len(list_train),a.scale_size,a.scale_size,3),dtype=np.uint8)    
    
    for i in range(len(list_train)):
        filename_in=list_train[i]
        try:
            filename_in=filename_in.replace('\n','')
        except:
            no=0
        
        IMg=io.imread(a.input_dir_all+'/Inputs/'+filename_in)
        Images_input[i,:,:,:]=IMg
        IMg=io.imread(a.input_dir_all+'/Targets_'+a.task_No+'/'+filename_in)
        Images_target[i,:,:,:]=IMg
#        plt.figure()
#        plt.imshow(IMg)
            
        
    X_train = PreProcess(Images_input) 
    del Images_input
    gc.collect()    
    
    Y_train = PreProcess(Images_target) 
    del Images_target
    gc.collect()    
    
#        plt.imshow(Y_train[0,:,:,:])
#       plt.imshow(X_train[0,:,:,:])
    
    print('============================')
    print('============================')
    print(datetime.datetime.now())
    print('============================')
    print('============================')
    
    a.checkpoint=a.output_dir    
    main()
    K.clear_session()
    
    a.mode="test"
    print('############',a.mode)
    tf.reset_default_graph()
    
    
    a.output_dir=a.output_dir_all+'/unet_ST/Results_t'+a.task_No+'/'+SetNames[cv]    
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    
    Images_input=np.zeros((len(list_test),a.scale_size,a.scale_size,3),dtype=np.uint8)    
    Images_target=np.zeros((len(list_test),a.scale_size,a.scale_size,3),dtype=np.uint8)    
    
    for i in range(len(list_test)):
        filename_in=list_train[i]
        try:
            filename_in=filename_in.replace('\n','')
        except:
            no=0
        
        IMg=io.imread(a.input_dir_all+'/Inputs/'+filename_in)
        Images_input[i,:,:,:]=IMg
        IMg=io.imread(a.input_dir_all+'/Targets_'+a.task_No+'/'+filename_in)
        Images_target[i,:,:,:]=IMg
        
        
#        plt.figure()
#        plt.imshow(IMg)
            
        
    X_test = PreProcess(Images_input) 
    del Images_input
    gc.collect()    
    
    Y_test = PreProcess(Images_target) 
    del Images_target
    gc.collect()                   
    main()
    K.clear_session()

    
    

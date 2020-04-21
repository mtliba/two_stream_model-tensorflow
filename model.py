from data_loader import import_train_data
import sys
from os import listdir, makedirs
from os.path import isfile, join, abspath, dirname
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt






class Two_stream_Model():
    def __init__(self, weights=''):
       
        self.build_Two_stream_Model()

        #load weights if provided
        if weights:
          self.model.load_weights(weights , by_name=True)
    
    # define vgg16 this one will be used in both stream 

    def build_vgg16(self, input_shape, stream_type):
        img_input = layers.Input(shape=input_shape, name='Input_' + stream_type)
        # Block 1
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv1_'+stream_type)(img_input)
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block1_conv2_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_'+stream_type)(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv1_'+stream_type)(x)
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          trainable=True,
                          name='block2_conv2_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_'+stream_type)(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1_'+stream_type)(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2_'+stream_type)(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_'+stream_type)(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3_'+stream_type)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_'+stream_type)(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1_'+stream_type)(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2_'+stream_type)(x)
        output = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3_'+stream_type)(x)

        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_'+stream_type)(output)

        model = Model(inputs=img_input, outputs=x)

        #initialize each vgg16 stream with ImageNet weights
        
        try:

          print(abspath(dirname(__file__)))
          model.load_weights(os.path.dirname(os.path.abspath(__file__))+'path to vgg weights in the same folder', by_name=False)
          model = Model(inputs=img_input, outputs=output)
          
          plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        
        except OSError:
          print(abspath(dirname(__file__)))
          print("ERROR: VGG weights are not found.")
          sys.exit(-1)
        return model.input, model.output


    def build_Two_stream_Model(self):
        #create two streams separately

        normal_stream_input, normal_stream_output = self.build_vgg16(input_shape=(400, 800, 3), stream_type='normal')
        scaled_stream_input , scaled_stream_output = self.build_vgg16(input_shape=(300, 400, 3), stream_type='rescaled')

        #add interpolation layer to the rescaled stream
        H,W = normal_stream_output.shape[1], normal_stream_output.shape[2]
        interp_layer = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(scaled_stream_output)
        assert interp_layer.shape[1] != H or interp_layer.shape[2] != W
        
        #add deconvolution network 
        # add hadmard prodcut layer with skip conx !!!!!!!!!!!!
        concat_layer = layers.concatenate([normal_stream_output, interp_layer])

        
        # change this layer to autoencuder bottelneck !!!!!!! lacke of data  leverage mean and dts in sampling 
        reshape_conv = layers.Conv2D(512, (1, 1),
                        activation='relu',
                        padding="same",
                        name='reshape_conv')(concat_layer)
        x = layers.Conv2DTranspose(
              512,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu',
             )(reshape_conv)                        

        x = layers.Conv2DTranspose(
              512,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)

        x = layers.Conv2DTranspose(
              512,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)
        
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)      
       
        x = layers.Conv2DTranspose(
              256,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)

        x = layers.Conv2DTranspose(
              256,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)
        x = layers.Conv2DTranspose(
              256,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = layers.Conv2DTranspose(
              128,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)
        x = layers.Conv2DTranspose(
              128,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x) 

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        x = layers.Conv2DTranspose(
              64,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x)
        x = layers.Conv2DTranspose(
              64,
              kernel_size=3,
              strides=(1, 1),
              padding="same",
              activation='relu')(x) 

        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        map_layer = layers.Conv2DTranspose(
              1,
              kernel_size=1,
              strides=(2, 2),
              activation='segmoid')(x)


        self.model = Model(inputs=[normal_stream_input, scaled_stream_input], outputs=map_layer)


        self.model.summary()
        
    # used only for test    
    def compute_saliency(self, img_path=None, img=None):
        data_mean = np.array([123, 116, 103])

        if img_path:
          img_normal = img_to_array(load_img(img_path,
              grayscale=False,
              target_size=(600, 800),
              interpolation='nearest'))

          img_scaled = img_to_array(load_img(img_path,
              grayscale=False,
              target_size=(300, 400),
              interpolation='nearest'))

        img_normal -= data_mean
        img_scaled -= data_mean

        img_normal = img_normal[None, :]/255
        img_scaled = img_scaled[None, :]/255

        smap = np.squeeze(self.model.predict([img_normal, img_scaled], batch_size=1, verbose=0))

        if img_path:
          img = cv2.imread(img_path)
          h, w = img.shape[:2]
        else:
          w, h = img.size[:2]

        smap = (smap - np.min(smap))/((np.max(smap)-np.min(smap)))
        smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)  
        smap = cv2.GaussianBlur(smap, (75, 75), 25, cv2.BORDER_DEFAULT) 

        return smap



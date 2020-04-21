from os import listdir, mkdir
from os.path import isfile, isdir, join
import numpy as np
import scipy.misc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from random import shuffle
import math


def import_train_data(dataset_root, image_dir, labels_dir ,input_size = (800,600) , use_cache=True):

    if use_cache:
        cache_dir = join(dataset_root, '__cache')
        if not isdir(cache_dir):
            mkdir(cache_dir)

        if isfile(join(cache_dir, 'data.npy')):
            image_data, rescaled_image_data, label_data = np.load(join(cache_dir, 'data.npy'))
            return image_data, rescaled_image_data, label_data

    image_names = [f for f in listdir(join(dataset_root, image_dir)) if isfile(join(dataset_root, image_dir , f))]
    
    shuffle(image_names)

    image_data = np.zeros((len(image_names), input_size[0],  input_size[1], 3), dtype=np.float32)
    rescaled_image_data = np.zeros((len(image_names), int(input_size[0]/2),  int(input_size[1]/2), 3), dtype=np.float32)
    label_data = np.zeros((len(image_names), input_size[0],  input_size[1], 1), dtype=np.float32)

    for i in range(len(image_names)):
        img_normal = img_to_array(load_img(join(dataset_root, image_dir, image_names[i]),
            grayscale=False,
            target_size= input_size,
            interpolation='nearest'))
        img_rescaled = img_to_array(load_img(join(dataset_root, image_dir, image_names[i]),
            grayscale=False,
            target_size= input_size/2,
            interpolation='nearest'))
        label = img_to_array(load_img(join(dataset_root, labels_dir, image_names[i]),
            grayscale=True,
            target_size= input_size,
            interpolation='nearest'))

        img_normal -= dataset_mean
        img_rescaled -= dataset_mean

        image_data[i] = img_normal[None, :]/255
        rescaled_image_data[i] = img_rescaled[None, :]/255
        label_data[i] = label[None, :]/255
        
    # create cache file contains numpy array in order to speed up next loading
    if use_cache:
        np.save(join(cache_dir, 'data.npy'), (image_data, rescaled_image_data, label_data))

    return image_data, rescaled_image_data, label_data



def train_data( data_dir = 'project/data/' , image_dir = 'images', label_dir = 'annotated_images' , val_prc = 0.1 , dataset_mean ,dataset_mean ):
   
    #assuming that the dataset structure is as follows
    #dataset_root
    #--> image_dir
    #--> label_dir
    # image_dir and label_dir should be provided as relative paths to dataset_root
    
    data_images = data_dir + image_dir
    print(" length of your dataset :", len(listdir(data_images)))
    # labled data shold be fixation point
    data_labels = data_dir + label_dir
    if len(listdir(data_images)) != len(listdir(data_labels)) :
        print("data unbalanssed !!!!! , missed images or labels")
    
    # import dataset 
    # by default all loaded data will be saved into cache
    # for train time we do not leverage this option
    # image data intended to feed into normal stream , rescaled image intended to fedd into rescaled stream

    image_data, rescaled_image_data, label_data = import_train_data(data_dir, image_dir, label_dir, input_size, use_cache=False)
    
    # specify percent of validaton set
    if float(val_prc) == 0.0 :
        num_val = int(math.floor(len(image_data)*0.1))
        val_index = np.random.choice(len(image_data), num_val, replace=False)
        train_index = np.setdiff1d(range(len(image_data)), val_index)

        X_train = [image_data[train_index], rescaled_image_data[train_index]]
        Y_train = label_data[train_index]

        X_val = [image_data[val_index], rescaled_image_data[val_index]]
        Y_val = label_data[val_index]

        return X_train, Y_train, X_val, Y_val      

    return [image_data, rescaled_image_data], label_data


import tensorflow as tf 
import matplotlib.pyplot as plt
import json as j
import numpy as np

'--- IMPORTING RAW DATA ---'

# meaningful variables 
training_data_directory='./data/training/rgb'
batch_size=64
image_size=[128,128]

def import_training_imgs(): # returns 
    return tf.keras.utils.image_dataset_from_directory(
        training_data_directory,label_mode=None, batch_size=batch_size, image_size = image_size, shuffle=False).map(lambda x: x/255)


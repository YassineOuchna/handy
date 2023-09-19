import tensorflow as tf 
import matplotlib.pyplot as plt
import json as j
import numpy as np

'--- IMPORTING RAW DATA ---'

# meaningful variables 
training_data_directory='./data/training/rgb'
batch_size=64
image_size=[128,128]

# turning raw images into 
dataset=tf.keras.utils.image_dataset_from_directory(
        training_data_directory,label_mode=None, batch_size=batch_size, image_size = image_size, shuffle=False).map(lambda x: x/255)

def visualize(dataset): # shows 2 elements of the dataset and prints out the shape
    plt.figure(figsize=(15, 15))
    for element in dataset:
        for i in range(2):
            ax=plt.subplot(2,1,i+1)
            plt.imshow(element[i])
            plt.axis('off')
            print(tf.shape(element[i]))
        break

visualize(dataset)
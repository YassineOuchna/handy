import tensorflow as tf
from tensorflow import keras
from keras import layers

"--- CONVOLUTION BLOCK ---"

class ConvBlock(tf.keras.Model):
    def __init__(self, out_dim):
        super(ConvBlock, self).__init__()
        self.core=keras.Sequential(
            layers.SpatialDropout2D(0.3),
            layers.Conv2D(out_dim, kernel_size=3,padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.3),
            layers.Conv2D(out_dim, kernel_size=3,padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.3),
            layers.Conv2D(out_dim, kernel_size=3,padding="same", activation="relu"),
        )
    
    
import tensorflow as tf
from tensorflow import keras
from keras import layers


"--- CONVOLUTION Block ---"

# We define the initial number of neurons
N = 16


class ConvBlock(tf.keras.Model):
    def __init__(self, out_dim):
        super(ConvBlock, self).__init__()
        self.core = keras.Sequential()
        self.core.add(layers.SpatialDropout2D(0.3))
        self.core.add(
            layers.Conv2D(out_dim, kernel_size=3,
                          padding="same", activation="relu"))
        self.core.add(
            layers.Conv2D(out_dim, kernel_size=3,
                          padding="same", activation="relu"))
        self.core.add(
            layers.SpatialDropout2D(0.3))
        self.core.add(
            layers.BatchNormalization())
        self.core.add(
            layers.SpatialDropout2D(0.3))
        self.core.add(
            layers.Conv2D(out_dim, kernel_size=3,
                          padding="same", activation="relu"))


'''---  BUILDING THE MODEL --- '''

model = ConvBlock(N)
for i in range(3):
    model.core.add(ConvBlock(N*(2)**(i+1)))

model.build((128, 128, 3))
model.summary()

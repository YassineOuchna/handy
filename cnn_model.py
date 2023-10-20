import tensorflow as tf
from tensorflow import keras
from keras import layers


"--- CONVOLUTION Block ---"

# We define the initial number of neurons
N = 16


def convblock(x, out_dim):
    # l0=layers.BatchNormalization()(x)
    # l1 = layers.SpatialDropout2D(0.3)(l0)
    # l2= layers.Conv2D(out_dim, kernel_size=3,
    #                     padding="same", activation="relu")(l1)
    l3 = layers.BatchNormalization()(x)
    l4 = layers.SpatialDropout2D(0.3)(l3)
    l5 = layers.Conv2D(out_dim, kernel_size=3,
                       padding="same", activation="relu")(l4)
    l6 = layers.BatchNormalization()(l5)
    l7 = layers.SpatialDropout2D(0.3)(l6)
    y = layers.Conv2D(out_dim, kernel_size=3,
                      padding="same", activation="relu")(l7)
    return y


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

# Input node
x = layers.Input(shape=(128, 128, 3))
model = convblock(x, N)
for i in range(3):
    model.core.add(ConvBlock(N*(2)**(i+1)))

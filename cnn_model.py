import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from data_processing import load_dataset, batch_size

"--- PARAMETERS ---"
learning_rate = 0.001
N = 16
dropout_rate = 0.3
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, epsilon=0.0001)
loss = tf.keras.losses.MeanSquaredError()
"--- Retrieving the dataset for training ---"

# training_dataset, validation_dataset = tf.keras.utils.split_dataset(full_dataset, left_size=0.8)
# For some reason, using the saved dataset doesn't work
# For now I am running the data processing from scratch

"--- CONVOLUTION Block ---"


def convblock(x, out_dim):
    # l0=layers.BatchNormalization()(x)
    # l1 = layers.SpatialDropout2D(0.3)(l0)
    # l2= layers.Conv2D(out_dim, kernel_size=3,
    #                     padding="same", activation="relu")(l1)
    l3 = layers.BatchNormalization()(x)
    l4 = layers.SpatialDropout2D(dropout_rate)(l3)
    l5 = layers.Conv2D(out_dim, kernel_size=3,
                       padding="same", activation="relu")(l4)
    l6 = layers.BatchNormalization()(l5)
    l7 = layers.SpatialDropout2D(dropout_rate)(l6)
    y = layers.Conv2D(out_dim, kernel_size=3,
                      padding="same", activation="relu")(l7)
    return y


'''---  BUILDING THE MODEL --- '''


def build_model():
    # Input node
    x = layers.Input(shape=(128, 128, 3), batch_size=batch_size)
    # Sequential blocks
    y_1 = convblock(x, N)
    y_1p = layers.MaxPool2D(pool_size=(2, 2))(y_1)
    y_2 = convblock(y_1p, 2*N)
    y_2p = layers.MaxPool2D(pool_size=(2, 2))(y_2)
    y_3 = convblock(y_2p, 4*N)
    y_3p = layers.MaxPool2D(pool_size=(2, 2))(y_3)
    y_4 = convblock(y_3p, 8*N)
    # Sequential blocks with skip connections to previous blocks
    y_4p = layers.UpSampling2D(size=(2, 2))(y_4)
    y_5 = convblock(tf.concat([y_3, y_4p], axis=3), 4*N)
    y_5p = layers.UpSampling2D(size=(2, 2))(y_5)
    y_6 = convblock(tf.concat([y_2, y_5p], axis=3), 2*N)
    y_6p = layers.UpSampling2D(size=(2, 2))(y_6)
    y_7 = convblock(tf.concat([y_1, y_6p], axis=3), N)
    # Final layer with post processing
    output = layers.Conv2D(21, activation='sigmoid',
                           kernel_size=3, padding='same')(y_7)

    return keras.Model(inputs=x, outputs=output)


def training(full_model, optimizer, loss):
    training_dataset = load_dataset()
    full_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=['accuracy'])
    full_model.fit(training_dataset, verbose=1, epochs=4)


# Visualizing input and prediction layer
def visualize(x, y):
    plt.figure(figsize=(2, 1))
    y = tf.reduce_sum(y, axis=-1)
    plt.imshow(y)
    plt.imshow(x)
    plt.show()


if __name__ == '__main__':
    model = build_model()
    model.summary()
    training(model, optimizer, loss)
    model.save('./model', overwrite=True)

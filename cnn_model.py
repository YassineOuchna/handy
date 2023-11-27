import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import data_processing

"--- PARAMETERS ---"
batch_size = 64
learning_rate = 0.01
N = 16
training_dataset_path = './data/dataset'
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.MeanSquaredError()
"--- Retrieving the dataset for training ---"

# training_dataset = tf.data.Dataset.load(training_dataset_path)

# For some reason, using the saved dataset doesn't work
# For now I am running the data processing from scratch

training_dataset = data_processing.load_dataset()

"--- CONVOLUTION Block ---"


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
    y_5 = convblock(tf.concat([y_4p, y_3], axis=3), 4*N)
    y_5p = layers.UpSampling2D(size=(2, 2))(y_5)
    y_6 = convblock(tf.concat([y_5p, y_2], axis=3), 2*N)
    y_6p = layers.UpSampling2D(size=(2, 2))(y_6)
    y_7 = convblock(tf.concat([y_6p, y_1], axis=3), N)
    # Final layer with post processing
    output = layers.Conv2D(21, activation='sigmoid',
                           kernel_size=3, padding='same')(y_7)

    return keras.Model(inputs=x, outputs=output)


def training(full_model, optimizer, loss):
    full_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=['accuracy'])
    full_model.fit(training_dataset)


# Visualizing input and prediction layer
def visualize(x, y):
    plt.figure(figsize=(2, 1))
    y = tf.reduce_sum(y, axis=-1)
    plt.imshow(y)
    plt.imshow(x)
    plt.show()


def print_info():
    print(training_dataset)
    i = 0
    for batch in training_dataset:
        print(tf.shape(batch[0]), tf.shape(batch[1]))
        break


if __name__ == '__main__':
    model = build_model()
    training(model, optimizer, loss)

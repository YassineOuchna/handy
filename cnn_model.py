import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

"--- PARAMETERS ---"
learning_rate = 1e-3
batch_size = 64
N = 16
training_dataset_path = './data/dataset'
"--- Retrieving the dataset for training ---"

training_dataset = tf.data.Dataset.load(training_dataset_path)

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

# Input node
x = layers.Input(shape=(128, 128, 3))
# Sequential blocks
y_1 = convblock(x, N)
y_2 = convblock(y_1, 2*N)
y_3 = convblock(y_2, 4*N)
y_4 = convblock(y_3, 8*N)
# Sequential blocks with skip connections to previous blocks
y_5 = convblock(tf.concat([y_4, y_3], axis=2), 4*N)
y_6 = convblock(tf.concat([y_5, y_2], axis=2), 2*N)
y_7 = convblock(tf.concat([y_6, y_1], axis=2), N)
# Final layer with post processing
output = layers.Conv2D(21, activation='sigmoid',
                       kernel_size=3, padding='same')(y_7)

full_model = keras.Model(inputs=x, outputs=output)


def training():
    full_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['accuracy', 'loss'])
    full_model.fit(training_dataset[0], training_dataset[1])


# Visualizing the output player
def visualize_out(y):
    plt.figure(figsize=(4, 6))
    for element in y:
        for i in range(4):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(element[i])
            plt.axis("off")
        plt.show()
        break


if __name__ == '__main__':
    visualize_out()

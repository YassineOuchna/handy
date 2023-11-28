import tensorflow as tf
import matplotlib.pyplot as plt
import json as j
import numpy as np
from tensorflow import keras


" Running this file visualizes the images, labels and first layer of a CNN and saves the final dataset"

"""---  IMPORTING RAW TRAINING DATA  ---"""


# Meaningful variables
training_data_directory = "./data/training/rgb"
batch_size = 32
image_size = [128, 128]

# Turning raw images into a keras tensor
# 130240 images of shape 128 x 128 x 3 (rgb values)
# that are put into batches of size 64
images_raw = keras.utils.image_dataset_from_directory(
    training_data_directory,
    label_mode=None,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
).map(lambda x: x / 255)

# Turning raw 3D labels into a 2D tensor

with open("./data/training_xyz.json", "r") as f:
    xyz_file = j.load(f)
    xyz_file = tf.convert_to_tensor(
        xyz_file
    )  # 3D Coordonates of 21 key points of the hand : 32560 x 21 x 3

with open("./data/training_K.json", "r") as k:
    K = j.load(k)
    K = tf.convert_to_tensor(K)  # 3D Coordonates of camera position


def visualize_img(images_raw):  # Shows 6 images of the dataset
    plt.figure(figsize=(10, 6))
    for element in images_raw:  # element representing one batch
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(element[i])
            plt.axis("off")
        plt.show()
        break


"""---  SETTING UP LABEL DATASET  ---"""


# Projection onto the 2D plane wiht linalg magic
uv = tf.transpose(
    tf.linalg.matmul(K, tf.transpose(xyz_file, perm=(0, 2, 1))), perm=(0, 2, 1)
)
label_raw = uv[:, :, :2] / uv[:, :, -1:]
# resizing the coordinates to a 128 x 128 shape
label_raw = label_raw * (127 / 224)

# for each label, we have 4 corresponding images
# having the same hand posture in different conditions

L1 = tf.concat([label_raw, label_raw], 0)
L2 = tf.concat([label_raw, label_raw], 0)


label_tensor = tf.concat([L1, L2], 0)  # tensor of shape 130240 x 21 x 2
# having the (x,y) of 21 hand posture defining points


# Creating a dataset with both images and labels with batches of size 64
label_origin = tf.data.Dataset.from_tensor_slices(label_tensor).batch(
    batch_size, drop_remainder=True
)
# dataset with x,y locations as labels
# dataset = tf.data.Dataset.zip((images_raw, label_origin))


def visualize_img_labels():       # shows the first 6 images with their corresponding labels
    plt.figure(figsize=(10, 10))
    for element in images_raw:
        for s in range(6):
            ax = plt.subplot(2, 3, s + 1)
            for i in range(5):
                X = [label_raw[s][k + i * 4][0] for k in range(1, 5)]
                Y = [label_raw[s][k + i * 4][1] for k in range(1, 5)]
                # inserting the coordinates of the root point 0
                X.insert(0, label_raw[s][0][0])
                Y.insert(0, label_raw[s][0][1])
                ax.plot(X, Y)
                plt.imshow(element[s])
        plt.show()
        break


"""---  CREATING A LABEL LAYER  ---"""

# from a set of points (raw label), creates an image of the label
# of the form 128 x 128 filled with ones and zeros


def create_label(label):
    # changing the type from float to int
    label = tf.cast(label, dtype=tf.int64)
    # removing points outside the 128 range
    label = tf.clip_by_value(label, 0, 127)
    batch_lvl = tf.concat(
        [tf.ones(21, dtype=tf.int64) * i for i in range(batch_size)], axis=0
    )
    point_lvl = tf.concat([tf.range(21, dtype=tf.int64)] * batch_size, axis=0)
    # loading x-axis values
    coordinates2 = tf.cast(label[:, :, 0], dtype=tf.int64)
    # loading y-axis values
    coordinates1 = tf.cast(label[:, :, 1], dtype=tf.int64)
    indices = tf.stack(
        [
            batch_lvl,
            tf.reshape(coordinates1, (batch_size * 21,)),
            tf.reshape(coordinates2, (batch_size * 21,)),
            point_lvl,
        ],
        axis=-1,
    )
    sparse_label = tf.sparse.SparseTensor(
        indices=indices,
        values=tf.ones(indices.shape[0]),
        dense_shape=(batch_size, 128, 128, 21),
    )
    x = tf.sparse.reorder(
        sparse_label
    )                                                          # Reordering dimensions
    not_sparse = tf.sparse.to_dense(x, validate_indices=False)
    return not_sparse

# Applies a gaussian blur on one image of a label
# returning an tensor image of shape 128 x 128 x 21


def __gaussian_blur(img, kernel_size=11, sigma=3):

    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(
            kernel[..., tf.newaxis], [1, 1, channels]
        )  # On ajoute les autres couleurs
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(
        img, gaussian_kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
    )


def create_layer(label):
    parsed_label = create_label(label)   # label image
    return __gaussian_blur(parsed_label)  # gauss'ed label image i.e layer


def load_dataset():
    layer_tensor = label_origin.map(create_layer)
    # Returns an image x layer dataset
    return tf.data.Dataset.zip((images_raw, layer_tensor))

# Shows 3 images with their corresponding heatmap of key points


def visualize_layer(data):
    plt.figure(figsize=(4, 6))
    for element in data:            # element being the particular batch
        for i in range(3):
            ax = plt.subplot(2, 3, i + 1)
            # Summing the 21 label values into one value per point
            image_label = tf.reduce_sum(element[1][i], axis=-1)
            print(element[1][i].shape)
            plt.imshow(image_label)
            ax = plt.subplot(2, 3, i+4)
            plt.imshow(element[0][i])
            # Recalculating the the coordinates from the heatmap
            coords = get_coordinates(element[1][i])
            x_val = []
            y_val = []
            for j in range(21):
                x_val.append(coords[j][0])
                y_val.append(coords[j][1])
            ax.plot(x_val, y_val)
            plt.axis("off")
        plt.show()
        break


"""---  SAVING THE FINAL DATASET  ---"""


def save_dataset(path):
    ds = load_dataset()
    ds.save(path)


def main():
    save_dataset('./data/dataset')


"""---  POST PROCESSING  ---"""

# Averging the output to find
# coordinates of different points


def get_coordinates(layer):  # layer being a heatmap of shape 128 x 128 x 21
    n = layer.shape[0]       # width and height
    feature_num = layer.shape[-1]
    indices_vector = []
    for k in range(n):
        indices_vector.append([k])
    indices_vector = tf.convert_to_tensor(indices_vector, dtype=tf.float32)
    indices_vector = tf.transpose(indices_vector)
    columns = tf.reduce_sum(layer, axis=1, keepdims=True)
    rows = tf.reduce_sum(layer, axis=0, keepdims=True)
    values = []
    for i in range(feature_num):
        x_values = tf.split(rows, num_or_size_splits=feature_num, axis=-1)[i]
        y_values = tf.split(
            columns, num_or_size_splits=feature_num, axis=-1)[i]
        average_x = tf.tensordot(indices_vector, x_values, axes=2)
        average_y = tf.tensordot(indices_vector, y_values, axes=2)
        values.append((float(average_x), float(average_y)))
    return values


if __name__ == "__main__":
    print("executed as main")

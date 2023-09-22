import tensorflow as tf
import matplotlib.pyplot as plt
import json as j
import numpy as np
from tensorflow import keras


"""---  IMPORTING RAW TRAINING DATA  ---"""


# Meaningful variables
training_data_directory = "./data/training/rgb"
batch_size = 64
image_size = [128, 128]

# Turning raw images into a keras tensor : 130240 images of shape 128 x 128 x 3 (rgb values)
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


# Projection onto the 2D plane wiht linalg voodoo
uv = tf.transpose(
    tf.linalg.matmul(K, tf.transpose(xyz_file, perm=(0, 2, 1))), perm=(0, 2, 1)
)
label_raw = uv[:, :, :2] / uv[:, :, -1:]
label_raw = label_raw * (127 / 224)  # resizing the coordinates to a 128 x 128 shape

# Augmented label tensor
L1 = tf.concat([label_raw, label_raw], 0)
L2 = tf.concat([label_raw, label_raw], 0)
label_tensor = tf.concat([L1, L2], 0)
label_tensor = label_tensor

# Creating the label dataset
label_origin = tf.data.Dataset.from_tensor_slices(label_tensor).batch(
    batch_size, drop_remainder=True
)
dataset = tf.data.Dataset.zip((images_raw, label_origin))


def visualize_img_labels():  # shows the first 6 images with their corresponding labels
    plt.figure(figsize=(10, 10))
    for element in images_raw:
        for s in range(6):
            ax = plt.subplot(2, 3, s + 1)
            for i in range(5):
                X = [label_raw[s][k + i * 4][0] for k in range(1, 5)]
                Y = [label_raw[s][k + i * 4][1] for k in range(1, 5)]
                X.insert(
                    0, label_raw[s][0][0]
                )  # inserting the coordinates of the root point 0
                Y.insert(0, label_raw[s][0][1])
                ax.plot(X, Y)
                plt.imshow(element[s])
        plt.show()
        break


"""---  CREATING A LABEL LAYER  ---"""


def create_label(label):
    label = tf.cast(
        label, dtype=tf.int64
    )  # On change le type du tenseur de float à int
    label = tf.clip_by_value(label, 0, 127)  # On écrete les valeurs extrèmes
    batch_lvl = tf.concat(
        [tf.ones(21, dtype=tf.int64) * i for i in range(batch_size)], axis=0
    )
    point_lvl = tf.concat([tf.range(21, dtype=tf.int64)] * batch_size, axis=0)
    coordinates2 = tf.cast(label[:, :, 0], dtype=tf.int64)  # On récupère les X
    coordinates1 = tf.cast(label[:, :, 1], dtype=tf.int64)  # On récupère les Y
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
    )  # on réordonne les dimension pour le bon fonctionnement de l'opération
    not_sparse = tf.sparse.to_dense(x, validate_indices=False)
    return not_sparse


def __gaussian_blur(img, kernel_size=11, sigma=3):
    """
    Applies a gaussian filter to one image
    """

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
    parsed_label = create_label(label)
    return __gaussian_blur(parsed_label)


def load_dataset():
    img_dataset = images_raw
    label_dataset = label_origin.map(create_layer)
    return tf.data.Dataset.zip((img_dataset, label_dataset))


if __name__ == "__main__":
    print("executed as main")
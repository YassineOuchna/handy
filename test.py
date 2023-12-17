import tensorflow as tf
import cv2
from data_processing import get_coordinates
import random as rd
import matplotlib.pyplot as plt
import numpy as np


def check_images():
    model = tf.keras.models.load_model('./model')
    images = []
    for i in range(6):
        k = rd.randint(10, 99)
        image = cv2.resize(cv2.imread(
            f'./data/training/rgb/000000{k}.jpg'), dsize=(128, 128))
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ = np.expand_dims(image, axis=0)
        heatmap = model(image_)[0]
        coords = get_coordinates(heatmap)
        # Drawing prediction on the image
        for i in range(5):
            points = []
            for k in range(1, 5):
                x = round(coords[i + k*4][0])
                y = round(coords[i + k*4][1])
                points.append((x, y))
            points.insert(0, (round(coords[0][0]), round(coords[0][1])))
            print(points)
            for p in range(len(points)-1):
                cv2.line(image, points[p], points[p+1], (0, 0, 255), 3)
        images.append(image)

    plt.figure(figsize=(4, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.show()


def train_test():
    pass


if __name__ == '__main__':
    check_images()

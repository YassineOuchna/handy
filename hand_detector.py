"File that uses the laptop's camera to detect in real time your hand posture"
import cv2
from data_processing import get_coordinates
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./model')
video = cv2.VideoCapture(0)

while True:

    # retrieving the frame in the forme
    # of a 480 x 640 x 3 array

    grabbed, frame = video.read()
    small_frame = cv2.resize(frame, (128, 128))
    small_frame = np.expand_dims(cv2.cvtColor(
        small_frame, cv2.COLOR_BGR2RGB), axis=0)
    heatmap = model(small_frame)[0]
    coords = get_coordinates(heatmap)
    for i in range(5):
        points = []
        for k in range(1, 5):
            x = round((coords[i + k*4][0])*(640/120))
            y = round((coords[i + k*4][1])*(480/120))
            points.append((x, y))
        points.insert(0, (round(coords[0][0]), round(coords[0][1])))
        print(points)
        for p in range(len(points)-1):
            cv2.line(frame, points[p], points[p+1], (0, 0, 255), 3)
    cv2.imshow('Hand Detector', frame)

    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()

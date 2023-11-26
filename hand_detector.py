"File that uses the laptop's camera to detect in real time your hand posture"
import cv2

video = cv2.VideoCapture(0)

while True:
    # retrieving the frame in the forme
    # of a 480 x 640 x 3 array
    grabbed, frame = video.read()
    cv2.imshow('Hand Detector', frame)
    frame = cv2.resize(frame, (128, 128))
    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()

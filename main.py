from deepface import DeepFace
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    emotion = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)

    help = emotion[0]

    txt = 'Emotion: ' + str(help['dominant_emotion'])

    cv2.putText(frame, txt, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()

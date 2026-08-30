import cv2
import cvzone
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("custom_data/c_f_s_a_2.MOV")
model = YOLO("yolov8n-pose.pt")


while True:
    ret, frame = cap.read() # ret is a boolean, frame is the video file

    if not ret: # if you video is done playing, wnat to play again
        cap = cv2.VideoCapture("custom_data/c_f_s_a_2.MOV")
        continue

    result = model(frame)
    frame = result[0].plot()
    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord("t"): # Loading each file per second, and  press "t" to stop playing
        break

cap.release()
cv2.destroyAllWindows()
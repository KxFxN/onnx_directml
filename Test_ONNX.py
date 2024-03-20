from yolov8 import ONNX 
import cv2
import time

prev_frame_time = 1
new_frame_time = 0

cap = cv2.VideoCapture('Video/Test1.mp4')

model = ONNX('Model/Model1.onnx', 0.6, 0.6)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    new_frame_time = time.time()
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    output = model(frame)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(frame,f'FPS:{int(fps)}',(20,40),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

    cv2.imshow("Detected Objects", output)     

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
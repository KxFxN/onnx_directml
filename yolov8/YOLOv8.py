import argparse
import onnxruntime as ort
import cv2
import numpy as np
import time
import sys

from yolov8.predict import Predict
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

class ONNX:

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.predict_model = Predict(onnx_model, confidence_thres, iou_thres)

    def __call__(self,image):
        self.results , self.images = self.predict_model.detect_objects(image)
        # return self.results
        
        if "for" in sys._getframe(1).f_code.co_name:
            return self.results
        else:
            return self
    
    def __str__(self):
        output_str = ""
        output_str += f"boxes: {self.predict_model.session.get_outputs()[0]}\n"  
        output_str += f"names: {self.predict_model.classes}\n" 
        output_str += f"orig_img: {self.images}\n"  
        output_str += f"orig_shape: {self.images.shape[:2]}\n"  
        output_str += f"speed: {self.predict_model.speed}ms\n"  
        return output_str
    
    def __iter__(self):
        self.iter_idx = 0
        return self


    def __next__(self):
        if self.iter_idx >= len(self.results):
            raise StopIteration

        # result = self.results[self.iter_idx] #สำหรับไม่ใช้ class Result เช่น print(results[0])
        result = Result(*self.results[self.iter_idx]) #สำหรับใช้ class Result เช่น print(results.box)
        self.iter_idx += 1
        return result

class Result:
    def __init__(self, box, score, class_name):
        self.box = box
        self.conf = score
        self.cls = class_name

    def __repr__(self):
        return f"(box={self.box}, conf={self.conf}, class_name={self.cls})"
    

def process_image(image_path, model):
    image = cv2.imread(image_path)
    output_image = model(image)
    return output_image

def process_video(video_path, model):
    prev_frame_time = 0
    new_frame_time = 0
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        new_frame_time = time.time()
        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        output = model(frame)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(frame,f'FPS:{int(fps)}',(20,40),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)

        cv2.imshow("Detected Objects", output)     

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    # parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg") if ASSETS else None, nargs='?', const=True, help="Path to input image.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    model = ONNX(args.model, args.conf, args.iou)

    if args.video:
        process_video(args.video, model)
    elif args.img:
        output_image = process_image(args.img, model)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


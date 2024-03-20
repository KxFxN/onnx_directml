import argparse
import onnxruntime as ort
import cv2
import numpy as np
import time


from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

class ONNX:

    def __init__(self, onnx_model, confidence_thres, iou_thres):

        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.initialize_model(onnx_model)

        # Load the class names from the COCO dataset
        # self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]
        self.classes = self.get_class_name()

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __call__(self, image):
        return self.detect_objects(image)
    
    def get_class_name(self):
        # รับข้อมูลเมตาดาต้าจากโมเดล
        metadata = self.session.get_modelmeta().custom_metadata_map['names']

        # แยก string ด้วย "',"
        split_metadata = metadata.split("', ")

        # สร้าง list เพื่อเก็บชื่อคลาส
        class_names = []

        for item in split_metadata:
            # แยก string ด้วย ":"
            parts = item.split(": ")

            # รับส่วนที่สองของ string ซึ่งเป็นชื่อคลาส
            class_name = parts[1].strip(" {}'")

            # เพิ่มชื่อคลาสลงใน list
            class_names.append(class_name)

        return class_names

    def initialize_model(self,path):
        self.session = ort.InferenceSession(path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = box.astype(int)

        color = self.color_palette[class_id]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Display class label and confidence score
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # label_y = y1 - label_height - 10 if y1 - label_height - 10 > 10 else y1 + label_height + 10

        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs
    
    def detect_objects(self,image):
        input_tensor = self.preprocess(image)

        outputs = self.inference(input_tensor)

        output = self.postprocess(image,outputs)

        return output

    def preprocess(self,image):
        self.img_height, self.img_width = image.shape[:2]

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame , (self.input_width,self.input_height))

        frame_data  = frame / 255.0
        frame_data  = frame_data.transpose(2,0,1)
        frame_data  = frame_data[np.newaxis, :, :, :].astype(np.float32)
        
        return frame_data
    
    def postprocess(self, input_image, output):
        outputs = np.squeeze(output[0]).T

        scores = np.max(outputs[:, 4:], axis=1)
        outputs = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]

        if len(scores) == 0:
            return input_image

        class_ids = np.argmax(outputs[:, 4:], axis=1)

        boxes = self.extract_boxes(outputs)

        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_thres)

        self.boxes_data = []

        for i in indices:
            box = boxes[i, :4]  # Extract x1, y1, x2, y2 from the box
            score = scores[i]
            class_id = class_ids[i]

            self.boxes_data.append((box, score, self.classes[class_id]))

            self.draw_detections(input_image, box, score, class_id)

        return input_image
    
    def xywh2xyxy(self,x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def intersection_over_union(self,boxA, boxB):
        # Calculate the intersection area
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Calculate the area of both boxes
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Calculate the intersection over union
        iou = inter_area / float(boxA_area + boxB_area - inter_area)

        return iou
    
    def non_max_suppression(self,boxes, scores, iou_threshold):
        indices = np.argsort(scores)[::-1]
        selected_indices = []

        while len(indices) > 0:
            current_index = indices[0]
            selected_indices.append(current_index)

            current_box = boxes[current_index]
            other_boxes = boxes[indices[1:]]

            iou_values = np.array([self.intersection_over_union(current_box, other_box) for other_box in other_boxes])

            indices = indices[np.where(iou_values <= iou_threshold)[0] + 1]

        return np.array(selected_indices)

    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):

        unique_class_ids = np.unique(class_ids)
        selected_indices = []

        for class_id in unique_class_ids:
            class_mask = (class_ids == class_id)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            indices = self.non_max_suppression(class_boxes, class_scores, iou_threshold)

            selected_indices.extend(np.where(class_mask)[0][indices])

        return np.array(selected_indices)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]    

    def verbose(self):
        log_string = ""
        if not self.boxes_data:
            log_string += "(no detections)"
        else:
            for box, score, class_name in self.boxes_data:
                log_string += f"{score:.2f} {class_name}, "
        return log_string.rstrip(", ")
   
    def boxes(self):
        return Boxes(self.boxes_data[0])

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

class Boxes:
    def __init__(self, boxes_data):
        self.box = boxes_data[0]
        self.conf = boxes_data[1]
        self.cls = boxes_data[2]

    def __repr__(self):
       return f"box:{self.box}\nconf:{self.conf}\ncls:{self.cls}"
    
    def __len__(self):
        return len([self.box,self.conf,self.cls])
    
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


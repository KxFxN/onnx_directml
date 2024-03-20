import argparse
import onnxruntime as ort
import cv2
import numpy as np
import time


from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

class Predict:

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.session = ort.InferenceSession(onnx_model, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.classes = self.get_class_name()
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(np.uint8)

    def detect_objects(self,image):
        input_tensor = self.preprocess(image)
        outputs = self.inference(input_tensor)
        postprocess = self.postprocess(image, outputs)

        total_time = self.preprocess_time + self.inference_time + self.postprocess_time

        # Store time information in a dictionary
        self.speed = {
            'preprocess': self.preprocess_time,
            'inference': self.inference_time,
            'postprocess': self.postprocess_time,
            'total': total_time
        }

        return postprocess
    

    def preprocess(self,image):
        start_speed = time.time()

        self.img_height, self.img_width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(image, (self.input_width, self.input_height))
        input_tensor = input_tensor.transpose(2, 0, 1) / 255.0
        input_tensor = input_tensor[np.newaxis, :, :, :].astype(np.float32)

        end_speed = time.time()
        self.preprocess_time = (end_speed - start_speed) * 1000

        return input_tensor

    def inference(self,input_tensor):
        start_speed = time.time()

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        end_speed = time.time()
        self.inference_time = (end_speed - start_speed) * 1000

        return outputs
    
    def postprocess(self,input_image, outputs):
        start_speed = time.time()

        outputs = np.squeeze(outputs[0]).T
        scores = np.max(outputs[:, 4:], axis=1)
        outputs = outputs[scores > self.confidence_thres, :]
        scores = scores[scores > self.confidence_thres]

        class_ids = np.argmax(outputs[:, 4:], axis=1)

        boxes = self.extract_boxes(outputs)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_thres)

        arry_box = []
        for i in indices:
            box = boxes[i, :4]
            score = scores[i]
            class_id = class_ids[i]

            arry_box.append((box, score, self.classes[class_id]))

            self.draw_detections(input_image, box, score, class_id)

        end_speed = time.time()
        self.postprocess_time = (end_speed - start_speed) * 1000
    
        return arry_box,input_image
    
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)
        return boxes
    
    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def intersection_over_union(self, boxA, boxB):
        # Calculate the intersection area
        xA = np.maximum(boxA[0, 0], boxB[0, 0])
        yA = np.maximum(boxA[0, 1], boxB[0, 1])
        xB = np.minimum(boxA[0, 2], boxB[0, 2])
        yB = np.minimum(boxA[0, 3], boxB[0, 3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Calculate the area of both boxes
        boxA_area = (boxA[0, 2] - boxA[0, 0] + 1) * (boxA[0, 3] - boxA[0, 1] + 1)
        boxB_area = (boxB[0, 2] - boxB[0, 0] + 1) * (boxB[0, 3] - boxB[0, 1] + 1)

        # Calculate the intersection over union
        iou = inter_area / float(boxA_area + boxB_area - inter_area)

        return iou

    def non_max_suppression(self, boxes, scores, iou_threshold):
        indices = np.argsort(scores)[::-1]
        selected_indices = []

        while len(indices) > 0:
            current_index = indices[0]
            selected_indices.append(current_index)

            current_box = boxes[current_index]
            other_boxes = boxes[indices[1:]]

            iou_values = np.array([self.intersection_over_union(current_box.reshape(1, 4), other_box.reshape(1, 4)) for other_box in other_boxes])
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

    def draw_detections(self, img, box, score, class_id):
        x1, y1, x2, y2 = box.astype(int)
        color = self.color_palette[class_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(color.tolist()), 2)

        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), tuple(color.tolist()), cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def get_class_name(self):
        metadata = self.session.get_modelmeta().custom_metadata_map['names']
        class_names = [item.split(": ")[1].strip(" {}'") for item in metadata.split("', ")]
        return class_names


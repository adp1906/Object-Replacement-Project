import cv2
import numpy as np

def load_yolov4_model():
    return cv2.dnn.readNet('yolov4.cfg', 'yolov4.weights')
                           
def load_coco_names():
    with open('coco.names', 'r') as f:
        return f.read().strip().split('\n')
                           
def perform_object_detection(net, image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    return net.forward(output_layers_names)

def draw_bounding_boxes(image, outs, classes):
    height, width, _ = image.shape
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
    return image
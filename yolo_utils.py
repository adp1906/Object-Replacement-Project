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

def draw_bounding_boxes(image, outs, classes, target_class='book'):
    height, width, _ = image.shape
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    print(f"indices: {indices}")
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")
    print(f"class_ids: {class_ids}")
    
    if len(indices) > 0:
        for idx in indices.flatten().astype(int):
            box = boxes[idx]
            x, y, w, h = box
            class_id = class_ids[idx]
            confidence = confidences[idx]
            
            if classes[class_id] == target_class:
                print(f"Class ID: {class_id}, Confidence: {confidence}")
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No indices found after non-max suppression.")
        
    print("End of Loop")
    
    return image
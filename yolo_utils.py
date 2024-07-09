import cv2
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)

def load_yolov4_model(cfg_path: str = 'yolov4.cfg', 
                      weights_path: str = 'yolov4.weights') -> cv2.dnn_Net:
    """
    Load YOLOv4 model from configuration and weights files.
    
    Parameters:
    cfg_path (str): Path to the YOLOv4 configuration file.
    weights_path (str): Path to the YOLOv4 weights file.
    
    Returns:
    cv2.dnn_Net: Loaded YOLOv4 network.
    """
    
    try:
        net = cv2.dnn.readNet(cfg_path, weights_path)
    except Exception as e:
        logging.error(f"Error loading YOLOv4 model from {cfg_path} and {weights_path}: {e}")
        raise IOError(f"Error loading YOLOv4 model from {cfg_path} and {weights_path}: {e}")
    return net
                           
def load_coco_names(file_path: str = 'coco.names') -> List[str]:
    """
    Load COCO class names from a file.
    
    Parameters:
    file_path (str): Path to the file containing COCO class names.
    
    Returns:
    List[str]: List of COCO class names.
    """
    
    try:
        with open(file_path, 'r') as f:
            class_names = f.read().strip().split('\n')
    except Exception as e:
        logging.error(f"Error loading COCO names from {file_path}: {e}")
        raise IOError(f"Error loading COCO names from {file_path}: {e}")
    return class_names
                           
def perform_object_detection(net: cv2.dnn_Net, 
                             image: np.ndarray, 
                             input_size: Tuple[int, int] = (416, 416), 
                             scalefactor: float = 1/255.0, 
                             swapRB: bool = True, 
                             crop: bool = False) -> List[np.ndarray]:
    """
    Perform object detection using YOLOv4.
    
    Parameters:
    net (cv2.dnn_Net): Loaded YOLOv4 network.
    image (np.ndarray): Input image for detection.
    input_size (tuple): Size to which the image will be resized.
    scalefactor (float): Scale factor for the image.
    swapRB (bool): Flag to swap the red and blue channels.
    crop (bool): Flag to crop the image.
    
    Returns:
    List[np.ndarray]: Output layers from the network after forwarding the input image.
    """
    
    blob = cv2.dnn.blobFromImage(image, scalefactor=scalefactor, size=input_size, swapRB=swapRB, crop=crop)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    return net.forward(output_layers_names)

def draw_bounding_boxes(image: np.ndarray, 
                        outs: List[np.ndarray], 
                        classes: List[str], 
                        target_class: str = 'book', 
                        conf_threshold: float = 0.5, 
                        nms_threshold: float = 0.4) -> np.ndarray:
    """
    Draw bounding boxes around detected objects.
    
    Parameters:
    image (np.ndarray): Input image.
    outs: Output layers from the network.
    classes (List[str]): List of class names.
    target_class (str): The class of object to highlight.
    conf_threshold (float): Confidence threshold.
    nms_threshold (float): Non-maximum suppression threshold.
    
    Returns:
    np.ndarray: Image with bounding boxes drawn.
    """
    
    height, width, _ = image.shape
    
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
    
    if len(indices) > 0:
        for idx in indices.flatten().astype(int):
            box = boxes[idx]
            x, y, w, h = box
            class_id = class_ids[idx]
            confidence = confidences[idx]
            
            if classes[class_id] == target_class:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{classes[class_id]}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        logging.info("No indices found after non-max suppression.")
    
    return image
import cv2
import logging

logging.basicConfig(level=logging.INFO)

def replace_object(image, outs, classes, target_class, replacement_image, conf_threshold=0.5):
    """
    Replace the detected object in an image with a replacement image.
    
    Parameters:
    image (np.ndarray): The original image.
    outs (list): Output layers from the YOLO detection model.
    classes (list): List of class names.
    target_class (str): The target class to replace.
    replacement_image (np.ndarray):The image to replace the detected object with.
    conf_threshold (float): Confidence threshold for detection.
    
    Returns:
    np.ndarray: The image with the detected object replaced.
    """
    
    height, width, _ = image.shape
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            
            if confidence > conf_threshold and classes[class_id] == target_class:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Ensure replacement image dimensions match the detected object.
                if replacement_image.shape[0] != h or replacement_image.shape[1] != w:
                    replacement_image = cv2.resize(replacement_image, (w, h), interpolation=cv2.INTER_CUBIC)
                
                try:
                    image[y:y+h, x:x+w] = replacement_image
                    logging.info(f"Replaced detected {target_class} with replacement image.")
                except Exception as e:
                    logging.error(f"Error replacing object: {e}")
                    continue
                
    return image
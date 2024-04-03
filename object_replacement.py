import cv2

def replace_object(image, outs, classes, target_class, replacement_image, conf_threshold=0.5):
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
                
                # Replace the detected object with replacement image
                image[y:y+h, x:x+w] = cv2.resize(replacement_image, (w, h), interpolation=cv2.INTER_CUBIC)
                
    return image
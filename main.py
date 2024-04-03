from yolo_utils import load_yolov4_model, load_coco_names, perform_object_detection, draw_bounding_boxes
from object_replacement import replace_object
import matplotlib.pyplot as plt
import cv2

def display_output_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
def main():
    # Load YOLOv4 model
    net = load_yolov4_model()
    
    # Load COCO class names
    classes = load_coco_names()
    
    # Read input image
    image = cv2.imread('harry_potter_desk.jpeg')
    
    # Get Image Dimensions
    height, width, _ = image.shape
    
    # Perform Object Detection
    outs = perform_object_detection(net, image)

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image.copy(), outs, classes)
    
    # Load Replacement Image
    replacement_image = cv2.imread('sorcerors-stone.jpg')
                
    # Define Parameters
    target_class = 'book'
    conf_threshold = 0.5

    # Replacement detected object with replacement image
    image_with_replacement = replace_object(image_with_boxes, outs, classes, target_class, replacement_image, conf_threshold)
    
    # Display the output image
    display_output_image(image_with_replacement)
    
if __name__ == "__main__":
    main()
    
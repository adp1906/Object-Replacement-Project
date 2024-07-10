from yolo_utils import load_yolov4_model, load_coco_names, perform_object_detection, draw_bounding_boxes
from object_replacement import replace_object
import matplotlib.pyplot as plt
import cv2

def display_image(image, title="image"):
    plt.figure()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    
def main():
    # Display original image
    image = cv2.imread('harry_potter_desk.jpeg')
    display_image(image, title="Original Image")
    
    # Load YOLOv4 model
    net = load_yolov4_model()
    
    # Load COCO class names
    classes = load_coco_names()
    
    # Get Image Dimensions
    height, width, _ = image.shape
    
    # Perform Object Detection
    outs = perform_object_detection(net, image)

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image.copy(), outs, classes)
    display_image(image_with_boxes, title="Image with Bounding Boxes")
    
    # Load Replacement Image
    replacement_image = cv2.imread('sorcerors-stone.jpg')
                
    # Define Parameters
    target_class = 'book'
    conf_threshold = 0.5

    # Replacement detected object with replacement image
    image_with_replacement = replace_object(image_with_boxes, outs, classes, target_class, replacement_image, conf_threshold)
    
    # Display the output image
    display_image(image_with_replacement, title="Image with Replacement")
    plt.show(block=True)
    
if __name__ == "__main__":
    main()
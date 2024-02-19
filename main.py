from yolo_utils import load_yolov4_model, load_coco_names, perform_object_detection, draw_bounding_boxes
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import cv2

def display_output_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
#     cv2_imshow(image)
#     cv2.imshow('Object Replacement', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
def main():
    # Load YOLOv4 model
    net = load_yolov4_model()
    
    # Load COCO class names
    classes = load_coco_names()
    print(f"CLASSES: {classes}")
    
    # Read input image
    image = cv2.imread('book_portrait.jpeg')
    
    # Perform Object Detection
    outs = perform_object_detection(net, image)

    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image.copy(), outs, classes)
    
    # Display the output image
    display_output_image(image_with_boxes)
    
if __name__ == "__main__":
    main()
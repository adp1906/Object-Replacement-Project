import pytest
import cv2
import numpy as np
from unittest.mock import patch, mock_open
import sys
import os

sys.path.append("..")

from yolo_utils import load_yolov4_model, load_coco_names, perform_object_detection, draw_bounding_boxes

@pytest.fixture(scope="module")
def yolo_model():
    # Fixture to load the YOLO model once per test session
    cfg_path = os.path.join("..", "yolov4.cfg")
    weights_path = os.path.join("..", "yolov4.weights")
    return load_yolov4_model(cfg_path, weights_path)

@pytest.fixture(scope="module")
def coco_names():
    # Fixture to load the COCO names once per test session
    with patch("builtins.open", 
               new_callable=mock_open, 
               read_data="person\nbicycle\ncar\nbook") as mock_file:
        yield load_coco_names(os.path.join("..", "coco.names"))

def test_load_yolov4_model(yolo_model):
    # Test if the model loads correctly
    assert isinstance(yolo_model, cv2.dnn_Net)
        
def test_load_coco_names(coco_names):
    # Test if the COCO names load correctly using the fixture
    assert isinstance(coco_names, list)
    assert len(coco_names) > 0
        
def test_perform_object_detection(yolo_model):
    # Mock a YOLOv4 model and an image for testing
    image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    try:
        outs = perform_object_detection(yolo_model, image)
        
        if isinstance(outs, tuple):
            outs = list(outs) 
            
        assert isinstance(outs, list)
        assert all(isinstance(out, np.ndarray) for out in outs)
        assert len(outs) > 0
    except Exception as e:
        pytest.fail(f"Object detection failed: {e}")
        
def test_draw_bounding_boxes(coco_names):
    # Mock an image, detection outputs, and class names
    image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    outs = [np.random.rand(1, 85).reshape(1, -1)]

    try:
        result_image = draw_bounding_boxes(image, outs, coco_names, target_class='book')
        assert isinstance(result_image, np.ndarray)
    except IndexError as e:
        print(f"IndexError details: {e}")
        pytest.fail(f"Drawing bounding boxes failed: {e}")
    except Exception as e:
        pytest.fail(f"Drawing bounding boxes failed: {e}")
        
if __name__ == "__main__":
    pytest.main()

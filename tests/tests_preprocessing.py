import os
import numpy as np
from PIL import Image
import pytest
from preprocess_data import resize_image, normalize_image

def test_resize_image():
    img = Image.new('RGB', (500, 500), color = 'red')
    resized_img = resize_image(img, (1024, 1024))
    assert resized_img.size == (1024, 1024)
    
def test_normalize_image():
    img = Image.new('RGB', (1024, 1024), color = 'red')
    normalized_img_np = normalize_image(img)
    assert normalized_img_np.min() >= -1.0
    assert normalized_img_np.max() <= 1.0
    
def test_preprocessing_pipeline(tmpdir):
    input_dir = tmpdir.mkdir("input")
    output_dir = tmpdir.mkdir("output")
    
    img = Image.new('RGB', (500, 500), color = 'red')
    img_path = os.path.join(input_dir, 'test.jpg')
    img.save(img_path)
    
    resize_image(img, (1024, 1024)).save(os.path.join(output_dir, 'test.jpg'))
    with Image.open(os.path.join(output_dir, 'test.jpg')) as img:
        normalized_img_np = normalize_image(img)
        img = Image.fromarray(((normalized_img_np + 1.0) * 127.5).astype(np.uint8))
        img.save(os.path.join(output_dir, 'test_normalized.jpg'))
        
    processed_img_path = os.path.join(output_dir, 'test_normalized.jpg')
    assert os.path.exists(processed_img_path)
    with Image.open(processed_img_path) as processed_img:
        assert processed_img.size == (1024, 1024)
        processed_img_np = np.array(processed_img).astype(np.float32)
        assert processed_img_np.min() >= 0
        assert processed_img_np.max() <= 255
        
if __name__ == "__main__":
    pytest.main()

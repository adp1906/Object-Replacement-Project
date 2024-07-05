import os
from PIL import Image
import numpy as np

def resize_image(image, size=(1024, 1024)):
    return image.resize(size, Image.LANCZOS)

def normalize_image(image):
    image_np = np.array(image).astype(np.float32)
    normalized_image_np = (image_np / 127.5) - 1.0
    return normalized_image_np

def process_image(input_path, output_path, size=(1024, 1024)):
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = resize_image(img, size)
            normalized_image_np = normalize_image(img) 
            img = Image.fromarray(((normalized_image_np + 1.0) * 127.5).astype(np.uint8))
            img.save(output_path, format='PNG')
            print(f"Processed and Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

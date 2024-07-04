import os
from PIL import Image
import numpy as np

def resize_image(image, size=(1024, 1024)):
    return image.resize(size, Image.LANCZOS)

def normalize_image(image):
    image_np = np.array(image).astype(np.float32)
    normalized_image_np = (image_np / 127.5) - 1.0
    return normalized_image_np

input_directory = '/Volumes/Samsung_T5/Private Object Replacement Project/Book Images'
output_directory = '/Volumes/Samsung_T5/Private Object Replacement Project/Processed Data/StyleGAN'
size = (1024, 1024)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
for filename in os.listdir(input_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            with Image.open(os.path.join(input_directory, filename)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = resize_image(img, size)
                normalized_image_np = normalize_image(img) 
                img = Image.fromarray(((normalized_image_np + 1.0) * 127.5).astype(np.uint8))
                img.save(os.path.join(output_directory, filename))
                print(f"Processed and Saved: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
                         
print("All new images have been resized and normalized.")

import os
from preprocess_data import process_image

input_directory = os.getenv('INPUT_DIRECTORY', '/default/input/path')
output_directory = os.getenv('OUTPUT_DIRECTORY', '/default/output/path')
size = (1024, 1024)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
for filename in os.listdir(input_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        process_image(input_path, output_path, size)
    
print("All new images have been resized and normalized.")

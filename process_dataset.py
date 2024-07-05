import os
from preprocess_data import process_image

input_directory = '/Volumes/Samsung_T5/Private Object Replacement Project/Book Images'
output_directory = '/Volumes/Samsung_T5/Private Object Replacement Project/Processed Data/StyleGAN/Processed Data'
size = (1024, 1024)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
for filename in os.listdir(input_directory):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        process_image(input_path, output_path, size)
    
print("All new images have been resized and normalized.")

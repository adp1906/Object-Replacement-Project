import os
from preprocess_data import process_image
from dotenv import load_dotenv

load_dotenv()

input_directory = os.getenv('INPUT_DIRECTORY')
output_directory = os.getenv('OUTPUT_DIRECTORY')
size = (1024, 1024)

# Print paths to verify they are read correctly
print(f"Input Directory: {input_directory}")
print(f"Output Directory: {output_directory}")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
for filename in os.listdir(input_directory):
    print(f"Processing file: {filename}")
    if filename.endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        print(f"Input Path: {input_path}")
        print(f"Output Path: {output_path}")
        process_image(input_path, output_path, size)
    
print("All new images have been resized and normalized.")

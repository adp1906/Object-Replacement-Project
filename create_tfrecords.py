import tensorflow as tf
import os
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
def serialize_example(image):
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'height': _float_feature(float(image.shape[0])),
        'width': _float_feature(float(image.shape[1])),
        'depth': _float_feature(float(image.shape[2]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
    
def create_tfrecords(output_file, image_dir):
    with tf.io.TFRecordWriter(output_file) as writer:
        for filename in os.listdir(image_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(image_dir, filename)
                
                print(f"Processing file: {filename}")
                print(f"Image Path: {image_path}")
                
                try: 
                    with Image.open(image_path) as img:
                        img_np = np.array(img)
                        example = serialize_example(img_np)
                        writer.write(example)
                        print(f"Successfully wrote {filename} to TFRecords.")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    
output_file = os.getenv('TFRECORDS_FILE')
image_dir = os.getenv('OUTPUT_DIRECTORY')
create_tfrecords(output_file, image_dir)

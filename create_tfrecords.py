import tensorflow as tf
import os
from PIL import Image
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
def serialize_example(image):
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'height': _float_feature([image.shape[0]]),
        'width': _float_feature([image.shape[1]]),
        'depth': _float_feature([image.shape[2]])
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
    
def create_tfrecords(output_file, image_dir):
    with tf.io.TFRecordWriter(output_file) as writer:
        for filename in os.listdir(image_dir):
            if filename.endswith(".png"):
                image_path = os.path.join(image_dir, filename)
                with  Image.open(image_path) as img:
                    img_np = np.array(img)
                    example = serialize_example(img_np)
                    writer.write(example)
                    
output_file = os.getenv('TFRECORDS_FILE', '/default/tfrecords/path')
image_dir = os.getenv('OUTPUT_DIRECTORY', '/default/output/path')
create_tfrecords(output_file, image_dir)

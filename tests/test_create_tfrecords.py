import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pytest
import sys

sys.path.append("..")

from create_tfrecords import create_tfrecords

def test_create_tfrecords(tmpdir):
    input_dir = tmpdir.mkdir("input")
    output_file = str(tmpdir.join("output.tfrecords"))
    
    for i in range(5):
        img = Image.new('RGB', (1024, 1024), color='red')
        img_path = os.path.join(input_dir, f'test_{i}.png')
        img.save(img_path)
        
    create_tfrecords(output_file, str(input_dir))
    
    assert os.path.exists(output_file)
    
    raw_dataset = tf.data.TFRecordDataset(output_file)
    
    for raw_record in raw_dataset.take(5):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        height = example.features.feature['height'].float_list.value[0]
        width = example.features.feature['width'].float_list.value[0]
        depth = example.features.feature['depth'].float_list.value[0]
        
        assert height == 1024
        assert width == 1024
        assert depth == 3
        
        img_raw = example.features.feature['image'].bytes_list.value[0]
        img_np = np.frombuffer(img_raw, dtype=np.uint8)
        img_np = img_np.reshape((int(height), int(width), int(depth)))
        
        assert img_np.shape == (1024, 1024, 3)
        assert np.all(img_np == np.array([255, 0, 0]))
        
if __name__ == "__main__":
    pytest.main()

import dnnlib
import dnnlib.tflib as tflib
from training import misc
from training import training_loop
import pretrained_networks

tflib.init_tf()

network_pkl = "gdrive:networks/stylegan2-ffhq-config-f.pkl"
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)

dataset_path = '/Volumes/Samsung_T5/Private Object Replacement Project/Processed Data/StyleGAN/TFRecords'

desc = 'fine-tuning-stylegan'
training_options = {
    'num_gpus': 1,
    'total_kimg': 25000,
    'mirror_augment': True,
    'drange_net': [-1, 1],
    'G_fmap_base': 8192,
    'D_fmap_base': 8192,
    'dataset_train': dataset_path,
    'resume_pkl': network_pkl,
}

training_loop.training_loop(**training_options)

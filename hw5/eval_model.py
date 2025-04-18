import tf_keras
import tensorflow as tf
from tf_keras import models, layers
import os
from gen_datasets_new import gen_datasets_and_save, load_datasets, DatasetType
from classifier import Classifier

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

home_dir = os.getenv("HOME")
raw_dataset_path = os.path.join(home_dir, "iot/datasets/mini_speech_commands")
processed_dataset_path = os.path.join(home_dir, "iot/datasets/processed_mini_speech_commands")

# Rename 'spec' to 'loaded_spec_info' to clarify it's just informational now
train, val, test, loaded_spec_info = load_datasets(processed_dataset_path, batch_size)
import tf_keras
import tensorflow as tf
from tf_keras import models, layers, metrics
import os
from gen_datasets_new import gen_datasets_and_save, load_datasets, DatasetType
from classifier import Classifier
import numpy as np # Import numpy
import json


model_name = '10d'
model = tf_keras.models.load_model(f'training/{model_name}.keras')

# get layers
layers = model.layers
for layer in layers:
    layer_name = layer.name
    weights = layer.get_weights()
    if not weights:
        #print(f"Layer {layer_name} has no weights.")
        continue
    weight_np = np.concat([np.array(w).flatten() for w in weights])
    #print('Weights shape:', weight_np.shape)
    max_val = np.max(np.abs(weight_np))
    thresh = 0.01 * max_val
    sparsity = np.mean(np.abs(weight_np) < thresh)
    print(f"Layer: {layer_name}, Sparsity: {sparsity:.2%}")
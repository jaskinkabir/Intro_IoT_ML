import tf_keras
import tensorflow as tf
from tf_keras import models, layers
import os
from gen_datasets_new import gen_datasets_and_save, load_datasets, DatasetType
from classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

home_dir = os.getenv("HOME")
raw_dataset_path = os.path.join(home_dir, "iot/datasets/mini_speech_commands")
processed_dataset_path = os.path.join(home_dir, "iot/datasets/processed_mini_speech_commands")

# Rename 'spec' to 'loaded_spec_info' to clarify it's just informational now
train, val, test, loaded_spec_info = load_datasets(processed_dataset_path, batch_size)

num_classes = 4 # Number of classes in the dataset
# Define the label list based on the dataset
silence_str = "_silence"
unknown_str = "_unknown"
label_list = [silence_str, unknown_str] + ['stop', 'go'] # Should be ['_silence', '_unknown', 'stop', 'go']


#train = None
if train is None:
    print("No datasets found. Generating datasets...")
    # Rename 'spec' to 'generated_spec'
    train, val, test, generated_spec = gen_datasets_and_save(
        dataset_type = DatasetType.MINI,
        commands= ['stop', 'go'],
        limit_positive_samples=False,
        max_wavs_per_command = {'stop': 50, 'go': 250},
        base_data_dir = raw_dataset_path,
        processed_data_dir= processed_dataset_path,
    )
    # Ensure datasets are batched if generated
    if batch_size:
        train = train.batch(batch_size).prefetch(AUTOTUNE)
        val = val.batch(batch_size).prefetch(AUTOTUNE)
        test = test.batch(batch_size).prefetch(AUTOTUNE)

# Get input shape directly from the train dataset's element_spec AFTER batching
# train.element_spec will be (TensorSpec(shape=(batch_size, height, width, channels), ...), TensorSpec(shape=(batch_size,), ...))
# We want the shape of a single spectrogram: (height, width, channels)
input_shape = train.element_spec[0].shape[1:]
print(f"Input shape determined from dataset spec: {input_shape}")

# Define the model
_model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, 5, activation='relu'),
    layers.Conv2D(64, 5, activation='relu'),
    #layers.Conv2D(64, 3, activation='relu'),
    # layers.MaxPooling2D(pool_size=(1,2)),
    # layers.BatchNormalization(),
    
    # layers.Conv2D(64, 3, activation='relu'),
    # layers.BatchNormalization(),
    
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.BatchNormalization(),

    # layers.Conv2D(256, 3, activation='relu'),
    # layers.BatchNormalization(),
    
    # layers.Conv2D(256, 3, activation='relu'),
    # layers.BatchNormalization(),
    
    layers.GlobalMaxPooling2D(),
    layers.Dense(4),
])

model = Classifier(
    model = _model,
    optimizer = tf_keras.optimizers.Adam(learning_rate=0.001),
    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy'],
)

#model.model.summary()
train_hist = model.fit(
    reset_best_acc = True,
    save_path = 'training/initial_model.keras',
    stop_patience = 50,
    fit_kwargs = {
        'x': train,
        'validation_data': test,
        'epochs': 30,
        'verbose': 1,
    },
)

#model.save('training/initial_model.keras')

# plot loss and accuracy curves and save figure as png
def plot_loss_accuracy(history, save_path):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot training & validation loss values
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
plot_loss_accuracy(train_hist, 'training/loss_accuracy.png')


#!/usr/bin/env python
# coding: utf-8

# ## Basic Steps
# 
# 1. Run a baseline training script to build a speech commands model.
# 2. Add in your custom word to the training and test/validation sets.
#    - Modify labels, shape of your output tensor in the model.
#    - Make sure that feature extractor for the model aligns with the feature extractor 
#      used in the arduino code.
# 3. Re-train model. => TF Model using floating-point numbers, that recognizes Google word and custom word.
# 4. Quantize the model and convert to TFlite. => keyword_model.tflite file
# 5. Convert tflite to .c file, using xxd => model_data.cc
# 6. Replace contents of existing micro_features_model.cpp with output of xxd.
# 
# All of the above steps are done in this notebook for the commands 'left', 'right'.
# 
# 7. In micro_speech.ino, modify micro_op_resolver (around line 80) to add any necessary operations (DIFF_FROM_LECTURE)
# 8. In micro_features_model_settings.h, modify kSilenceIndex and kUnknownIndex, depending on 
# where you have them in commands.  
#   - Commands = ['left', 'right', '_silence', '_unknown'] => kSilenceIndex=2, kUnknownIndex=3
# 9. In micro_features_model_settings.cpp, modify kCategoryLabels to correspond to commands in this script.
# 10. In micro_features_micro_model_settings.h, set kFeatureSliceDurationMs, kFeatureSliceStrideMs to match what is passed to microfrontend as window_size, window_step, respectively.
# 11. Rebuild Arduino program, run it, recognize the two target words.
# 12. Experiment with model architecture, training parameters/methods, augmentation, more data-gathering, etc.
# 

# You can download the data set with this command line (or just a browser):
# 
# `wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz `

# In[1]:


def in_notebook():
  """
  Test if we are in a python script vs jupyter notebook.
  Returns True if called in a jupyter notebook, false if called in a standard python file
  taken from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
  """
  import __main__ as main
  return not hasattr(main, '__file__')


# In[2]:


# TensorFlow and tf_keras
import tensorflow as tf
import tf_keras

from tf_keras import Input, layers
from tf_keras import models
from tf_keras.layers.experimental import preprocessing
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

print(tf.__version__)

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if in_notebook():
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm 

import os, glob, pathlib, time, random

from datetime import datetime as dt
from IPython import display
import platform


# In[3]:


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# In[4]:


APPLE_SILICON = platform.processor() == 'arm'
# Define range of 16-bit integers
i16min = -2**15
i16max = 2**15-1
fsamp = 16000 # sampling rate
wave_length_ms = 1000 # 1000 => 1 sec of audio
wave_length_samps = int(wave_length_ms*fsamp/1000)

# you can change these next three
window_size_ms=64 
window_step_ms=48
num_filters = 32
batch_size = 32
use_microfrontend = True # recommended, but you can use another feature extractor if you like

## uncomment exactly one of these 
dataset = 'mini-speech'
#dataset = 'full-speech-files' # use the full speech commands stored as files 

commands = ['stop', 'go'] ## Change this line for your custom keywords

# limit the instances of each command in the training set to simulate limited data
limit_positive_samples = False
max_wavs_0 = 50  # use no more than ~ samples of commands[0]
max_wavs_1 = 250  # use no more than ~ samples of commands[1]

silence_str = "_silence"  # label for <no speech detected>
unknown_str = "_unknown"  # label for <speech detected but not one of the target words>
EPOCHS = 25

print(f"FFT window length = {int(window_size_ms * fsamp / 1000)}")

might_be = {True:"IS", False:"IS NOT"} # useful for formatting conditional sentences


# Apply the frontend to an example signal.

# In[44]:

home_dir = os.getenv("HOME")
processed_data_dir = os.path.join(home_dir, 'data', 'processed_speech_commands') # Directory to save/load processed datasets

if dataset == 'mini-speech':
  data_dir = pathlib.Path(os.path.join(home_dir, 'data', 'mini_speech_commands'))
  if not data_dir.exists():
    tf_keras.utils.get_file('mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True, 
      cache_dir=home_dir,
      cache_subdir='data'
    )
  # commands = np.array(tf.io.gfile.listdir(str(data_dir))) # if you want to use all the command words
  # commands = commands[commands != 'README.md']
elif dataset == 'full-speech-files':
  data_dir = pathlib.Path(os.path.join(os.getenv("HOME"), 'data', 'speech_commands_v0.02'))
  if not data_dir.exists():
    raise RuntimeError("Either download the speech commands files to {data_dir} or change this code to where you have them")
else:
  raise RuntimeError('dataset should either be "mini-speech" or "full-speech-files"')


# In[7]:


label_list = commands.copy()
label_list.insert(0, silence_str)
label_list.insert(1, unknown_str)
print('label_list:', label_list)


# In[8]:


if dataset in ['mini-speech', 'full-speech-files']:
    # filenames = tf.io.gfile.glob(str(data_dir) + '/*/*.wav') 
    # filenames = tf.io.gfile.glob(str(data_dir) + os.sep + '*' + '/' + '*.wav') 
    filenames = glob.glob(os.path.join(str(data_dir), '*', '*.wav')) 
  
    # with the next commented-out line, you can choose only files for words in label_list
    # filenames = tf.concat([tf.io.gfile.glob(str(data_dir) + '/' + cmd + '/*') for cmd in label_list], 0)
    random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    # print('Number of examples per label:',
    #       len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])


# In[9]:


# Not really necessary, but just look at a few of the files to make sure that 
# they're the correct files, shuffled, etc.

# for i in range(10):
#     print(filenames[i].numpy().decode('utf8'))


# In[10]:


if dataset == 'mini-speech':
  print('Using mini-speech')
  num_train_files = int(0.8*num_samples) 
  num_val_files = int(0.1*num_samples) 
  num_test_files = num_samples - num_train_files - num_val_files
  train_files = filenames[:num_train_files]
  val_files = filenames[num_train_files: num_train_files + num_val_files]
  test_files = filenames[-num_test_files:]
elif dataset == 'full-speech-files':  
  # the full speech-commands set lists which files are to be used
  # as test and validation data; train with everything else
  fname_val_files = os.path.join(data_dir, 'validation_list.txt')
  with open(fname_val_files) as fpi_val:
    val_files = fpi_val.read().splitlines()
  # validation_list.txt only lists partial paths
  val_files = [os.path.join(data_dir, fn) for fn in val_files]
  fname_test_files = os.path.join(data_dir, 'testing_list.txt')
  with open(fname_test_files) as fpi_tst:
    test_files = fpi_tst.read().splitlines()
  # testing_list.txt only lists partial paths
  test_files = [os.path.join(data_dir, fn).rstrip() for fn in test_files]    
    
  if os.sep != '/': 
    # the files validation_list.txt and testing_list.txt use '/' as path separator
    # if we're on a windows machine, replace the '/' with the correct separator
    val_files = [fn.replace('/', os.sep) for fn in val_files]
    test_files = [fn.replace('/', os.sep) for fn in test_files] 

  # convert the TF tensor filenames into an array of strings so we can use basic python constructs
  # train_files = [f.decode('utf8') for f in filenames.numpy()]

  
  # don't train with the _background_noise_ files; exclude when directory name starts with '_'
  train_files = [f for f in filenames if f.split(os.sep)[-2][0] != '_']
  # validation and test files are listed explicitly in *_list.txt; train with everything else
  train_files = list(set(train_files) - set(test_files) - set(val_files))
   
elif dataset == 'full-speech-ds':  
    print("Using full-speech-ds. This is in progress.  Good luck!")
else:
  raise ValueError("dataset must be either full-speech-files, full-speech-ds or mini-speech")
print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


# In[11]:


## Remove some of the target words if we're experimenting with limited data

if limit_positive_samples:
  if dataset not in ['mini-speech', 'full-speech-files']:
    raise RuntimeError("Right now, limit_positive_samples is only implemented if drawing the data from files")
  num_files_cmd0 = 0
  num_files_cmd1 = 0
  # elements of train_files look like this:
  # '/path/to/data/speech_commands_0_2_root/right/196e84b7_nohash_0.wav'
  # so if we split on '/' (or '\' in windows), the 2nd to last element is the label
  for idx,f in enumerate(train_files):
    if f.split(os.sep)[-2] == commands[0]:
      if num_files_cmd0 >= max_wavs_0:
        train_files.pop(idx)
      else:
        num_files_cmd0 += 1
    elif f.split(os.sep)[-2] == commands[1]:
      if num_files_cmd1 >= max_wavs_1:
        train_files.pop(idx)
      else:
        num_files_cmd1 += 1


# In[12]:


print(train_files[:5])
print(val_files[:5])
print(test_files[:5])


# In[13]:


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)


# In[14]:


def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  in_set = tf.reduce_any(parts[-2] == label_list)
  label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(unknown_str))
  # print(f"parts[-2] = {parts[-2]}, in_set = {in_set}, label = {label}")
  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return  label # parts[-2]


# In[15]:


def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label


# In[16]:


def get_spectrogram(waveform):
  # Concatenate audio with padding so that all audio clips will be of the 
  # same length (16000 samples)
  zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.int16)
  waveform = tf.cast(0.5*waveform*(i16max-i16min), tf.int16)  # scale float [-1,+1]=>INT16
  equal_length = tf.concat([waveform, zero_padding], 0)
  ## Make sure these labels correspond to those used in micro_features_micro_features_generator.cpp
  spectrogram = frontend_op.audio_microfrontend(equal_length, sample_rate=fsamp, num_channels=num_filters,
                                    window_size=window_size_ms, window_step=window_step_ms)
  return spectrogram


# Function to convert each waveform in a set into a spectrogram, then convert those
# back into a dataset using `from_tensor_slices`.  (We should be able to use 
# `wav_ds.map(get_spectrogram_and_label_id)`, but there is a problem with that process).
#    

# In[17]:


def create_silence_dataset(num_waves, samples_per_wave, rms_noise_range=[0.01,0.2], silent_label=silence_str):
    # create num_waves waveforms of white gaussian noise, with rms level drawn from rms_noise_range
    # to act as the "silence" dataset
    rng = np.random.default_rng()
    rms_noise_levels = rng.uniform(low=rms_noise_range[0], high=rms_noise_range[1], size=num_waves)
    rand_waves = np.zeros((num_waves, samples_per_wave), dtype=np.float32) # pre-allocate memory
    for i in range(num_waves):
        rand_waves[i,:] = rms_noise_levels[i]*rng.standard_normal(samples_per_wave)
    labels = [silent_label]*num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))  


# In[18]:


def wavds2specds(waveform_ds, verbose=True):
  wav, label = next(waveform_ds.as_numpy_iterator())
  one_spec = get_spectrogram(wav)
  one_spec = tf.expand_dims(one_spec, axis=0)  # add a 'batch' dimension at the front
  one_spec = tf.expand_dims(one_spec, axis=-1) # add a singleton 'channel' dimension at the back    

  num_waves = 0 # count the waveforms so we can allocate the memory
  for wav, label in waveform_ds:
    num_waves += 1
  print(f"About to create spectrograms from {num_waves} waves")
  spec_shape = (num_waves,) + one_spec.shape[1:] 
  spec_grams = np.nan * np.zeros(spec_shape)  # allocate memory
  labels = np.nan * np.zeros(num_waves)
  idx = 0
  for wav, label in waveform_ds:    
    if verbose and idx % 250 == 0:
      print(f"\r {idx} wavs processed", end='')
    spectrogram = get_spectrogram(wav)
    # TF conv layer expect inputs structured as 4D (batch_size, height, width, channels)
    # the microfrontend returns 2D tensors (freq, time), so we need to 
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # add a 'batch' dimension at the front
    spectrogram = tf.expand_dims(spectrogram, axis=-1) # add a singleton 'channel' dimension at the back
    spec_grams[idx, ...] = spectrogram
    new_label = label.numpy().decode('utf8')
    new_label_id = np.argmax(new_label == np.array(label_list))    
    labels[idx] = new_label_id # for numeric labels
    # labels.append(new_label) # for string labels
    idx += 1
  labels = np.array(labels, dtype=int)
  output_ds = tf.data.Dataset.from_tensor_slices((spec_grams, labels))  
  return output_ds


# In[19]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
num_train_files = len(train_files)
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
train_ds = wavds2specds(waveform_ds)


# In[20]:


def copy_with_noise(ds_input, rms_level=0.25):
  rng = tf.random.Generator.from_seed(1234)
  wave_shape = tf.constant((wave_length_samps,))
  def add_noise(waveform, label):
    noise = rms_level*rng.normal(shape=wave_shape)
    zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)    
    noisy_wave = waveform + noise
    return noisy_wave, label

  return ds_input.map(add_noise)


# In[21]:


def pad_16000(waveform, label):
    zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)        
    return waveform, label


# In[22]:


def count_labels(dataset):
    counts = {}
    for sample in dataset: # sample will be a tuple: (input, label) or (input, label, weight)
        lbl = sample[1]
        if lbl.dtype == tf.string:
            label = lbl.numpy().decode('utf-8')
        else:
            label = lbl.numpy()
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


# In[23]:


def is_batched(ds):
    ## This is probably not very robust
    try:
        ds.unbatch()  # does not actually change ds. For that we would ds=ds.unbatch()
    except:
        return False # we'll assume that the error on unbatching is because the ds is not batched.
    else:
        return True  # if we were able to unbatch it then it must have been batched (??)


# In[24]:


# Collect what we did to generate the training dataset into a 
# function, so we can repeat with the validation and test sets.
def preprocess_dataset(files, commands, label_list, silence_str, wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms, num_silent=None, noisy_reps_of_known=None):
  """
  Preprocesses a list of audio files into a spectrogram dataset.

  files -- list of files
  commands -- list of target command words
  label_list -- list of all possible labels (including silence, unknown)
  silence_str -- string label for silence
  wave_length_samps -- number of samples for 1 second audio
  fsamp -- sampling frequency
  num_filters -- number of mel filters for spectrogram
  window_size_ms -- window size for STFT
  window_step_ms -- window step for STFT
  num_silent -- number of silence samples to add
  noisy_reps_of_known -- list of rms noise levels for data augmentation

  Returns: a tf.data.Dataset of (spectrogram, label_id) tuples
  """
  if num_silent is None:
    num_silent = int(0.2*len(files))+1
  print(f"Processing {len(files)} files")
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  # Use local get_waveform_and_label that captures label_list and unknown_str from the outer scope
  def _get_waveform_and_label(file_path):
      label = get_label(file_path) # get_label needs access to label_list and unknown_str
      audio_binary = tf.io.read_file(file_path)
      waveform = decode_audio(audio_binary)
      return waveform, label

  waveform_ds = files_ds.map(_get_waveform_and_label, num_parallel_calls=AUTOTUNE)

  if noisy_reps_of_known is not None:
    # create a few copies of only the target words to balance the distribution
    # create a tmp dataset with only the target words
    ds_only_cmds = waveform_ds.filter(lambda w,l: tf.reduce_any(l == commands))
    num_added_noisy = 0
    for noise_level in noisy_reps_of_known:
       waveform_ds = waveform_ds.concatenate(copy_with_noise(ds_only_cmds, rms_level=noise_level))
       # This count isn't perfect as filter is lazy, but gives an idea
       num_added_noisy += tf.data.experimental.cardinality(ds_only_cmds).numpy() if tf.data.experimental.cardinality(ds_only_cmds) > 0 else 0


  if num_silent > 0:
    silent_wave_ds = create_silence_dataset(num_silent, wave_length_samps,
                                            rms_noise_range=[0.01,0.2],
                                            silent_label=silence_str)
    waveform_ds = waveform_ds.concatenate(silent_wave_ds)

  print(f"Added {num_silent} silent wavs" + (f" and {num_added_noisy} noisy command wavs" if noisy_reps_of_known else ""))

  # Use local get_spectrogram that captures necessary params from outer scope
  def _get_spectrogram(waveform):
      zero_padding = tf.zeros([wave_length_samps] - tf.shape(waveform), dtype=tf.int16)
      waveform = tf.cast(0.5*waveform*(i16max-i16min), tf.int16)  # scale float [-1,+1]=>INT16
      equal_length = tf.concat([waveform, zero_padding], 0)
      spectrogram = frontend_op.audio_microfrontend(equal_length, sample_rate=fsamp, num_channels=num_filters,
                                        window_size=window_size_ms, window_step=window_step_ms)
      return spectrogram

  # Use local wavds2specds that captures necessary params
  def _wavds2specds(local_waveform_ds, verbose=True):
      wav, label = next(local_waveform_ds.as_numpy_iterator())
      one_spec = _get_spectrogram(wav)
      one_spec = tf.expand_dims(one_spec, axis=0)  # add a 'batch' dimension at the front
      one_spec = tf.expand_dims(one_spec, axis=-1) # add a singleton 'channel' dimension at the back

      num_waves = tf.data.experimental.cardinality(local_waveform_ds)
      if num_waves == tf.data.experimental.UNKNOWN_CARDINALITY:
          print("Warning: Unknown dataset cardinality, iterating to count...")
          num_waves = 0
          for _ in local_waveform_ds:
              num_waves += 1
      print(f"About to create spectrograms from {num_waves} waves")

      spec_shape = (num_waves,) + one_spec.shape[1:]
      spec_grams = np.nan * np.zeros(spec_shape, dtype=np.float32)  # Use float32 for spectrograms
      labels = np.nan * np.zeros(num_waves, dtype=np.int64) # Use int64 for labels expected by SparseCategoricalCrossentropy
      idx = 0
      # Need to iterate through the original waveform_ds again as the iterator was consumed by next() or cardinality()
      for wav, label in tqdm(local_waveform_ds, total=num_waves, desc="Generating Spectrograms"):
          spectrogram = _get_spectrogram(wav)
          # TF conv layer expect inputs structured as 4D (batch_size, height, width, channels)
          spectrogram = tf.expand_dims(spectrogram, axis=-1) # add channel dim
          spec_grams[idx, ...] = spectrogram.numpy() # Store numpy array
          new_label = label.numpy().decode('utf8')
          new_label_id = np.argmax(new_label == np.array(label_list))
          labels[idx] = new_label_id
          idx += 1

      output_ds = tf.data.Dataset.from_tensor_slices((spec_grams, labels))
      return output_ds

  output_ds = _wavds2specds(waveform_ds)
  return output_ds


def preprocess_and_save_datasets(train_files, val_files, test_files, save_dir, commands, label_list, silence_str, wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms, noisy_reps_of_known=None):
    """Processes and saves the train, validation, and test datasets."""
    print("Preprocessing and saving datasets...")
    t_start = time.time()
    # Create base save directory if it doesn't exist
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Process and save Training set
    print("\n--- Processing Training Set ---")
    train_ds = preprocess_dataset(train_files, commands, label_list, silence_str, wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms, noisy_reps_of_known=noisy_reps_of_known)
    train_save_path = os.path.join(save_dir, 'train')
    tf.data.Dataset.save(train_ds, train_save_path)
    print(f"Training dataset saved to {train_save_path}")
    # Save element spec for robust loading later
    element_spec = train_ds.element_spec
    spec_save_path = os.path.join(save_dir, 'element_spec.pkl')
    import pickle
    with open(spec_save_path, 'wb') as f:
        pickle.dump(element_spec, f)
    print(f"Element spec saved to {spec_save_path}")


    # Process and save Validation set
    print("\n--- Processing Validation Set ---")
    val_ds = preprocess_dataset(val_files, commands, label_list, silence_str, wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms)
    val_save_path = os.path.join(save_dir, 'val')
    tf.data.Dataset.save(val_ds, val_save_path)
    print(f"Validation dataset saved to {val_save_path}")

    # Process and save Test set
    print("\n--- Processing Test Set ---")
    test_ds = preprocess_dataset(test_files, commands, label_list, silence_str, wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms)
    test_save_path = os.path.join(save_dir, 'test')
    tf.data.Dataset.save(test_ds, test_save_path)
    print(f"Test dataset saved to {test_save_path}")

    t_end = time.time()
    print(f"Dataset preprocessing and saving finished in {t_end - t_start:.2f} seconds.")
    return element_spec # Return spec for immediate use if needed

def load_processed_datasets(save_dir, batch_size):
    """Loads preprocessed datasets from disk."""
    print(f"Loading processed datasets from {save_dir}...")
    t_start = time.time()
    # Load element spec
    spec_save_path = os.path.join(save_dir, 'element_spec.pkl')
    import pickle
    try:
        with open(spec_save_path, 'rb') as f:
            element_spec = pickle.load(f)
        print("Loaded element spec.")
    except FileNotFoundError:
        print(f"Error: Element spec file not found at {spec_save_path}. Cannot load datasets.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading element spec: {e}. Cannot load datasets.")
        return None, None, None, None


    train_save_path = os.path.join(save_dir, 'train')
    val_save_path = os.path.join(save_dir, 'val')
    test_save_path = os.path.join(save_dir, 'test')

    try:
        train_ds = tf.data.Dataset.load(train_save_path, element_spec=element_spec)
        val_ds = tf.data.Dataset.load(val_save_path, element_spec=element_spec)
        test_ds = tf.data.Dataset.load(test_save_path, element_spec=element_spec)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure the saved datasets exist and the element spec is correct.")
        return None, None, None, None

    # Apply post-loading transformations
    train_ds = train_ds.shuffle(10000) # Adjust buffer size as needed
    val_ds = val_ds.shuffle(1000)   # Adjust buffer size as needed
    # No need to shuffle test_ds usually, but doesn't hurt

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size) # Batch test set for efficient evaluation

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE) # Prefetch test set

    t_end = time.time()
    print(f"Dataset loading and preparation finished in {t_end - t_start:.2f} seconds.")

    return train_ds, val_ds, test_ds, element_spec


# In[25]:


print(f"We have {len(train_files)}/{len(val_files)}/{len(test_files)} training/validation/test files")


# In[26]:


# train_files = train_files[0:10000] for a quick test run


# In[27]:


# Check if preprocessed data exists
train_save_path = os.path.join(processed_data_dir, 'train')
val_save_path = os.path.join(processed_data_dir, 'val')
test_save_path = os.path.join(processed_data_dir, 'test')
spec_save_path = os.path.join(processed_data_dir, 'element_spec.pkl')

if os.path.exists(train_save_path) and os.path.exists(val_save_path) and os.path.exists(test_save_path) and os.path.exists(spec_save_path):
    print(f"Found preprocessed data in {processed_data_dir}. Loading...")
    train_ds, val_ds, test_ds, element_spec = load_processed_datasets(processed_data_dir, batch_size)
    if train_ds is None:
        print("Failed to load datasets. Exiting.")
        exit() # Or handle error appropriately
    # Get input shape from the loaded element_spec
    input_shape = element_spec[0].shape
    print(f"Loaded datasets. Input shape: {input_shape}")

else:
    print(f"Preprocessed data not found in {processed_data_dir}. Processing and saving...")
    # Define noise levels for augmentation if needed
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, .1, .1, .1] # Example levels
    element_spec = preprocess_and_save_datasets(
        train_files, val_files, test_files, processed_data_dir,
        commands, label_list, silence_str, wave_length_samps, fsamp,
        num_filters, window_size_ms, window_step_ms,
        noisy_reps_of_known=noise_levels
    )
    print("Finished processing and saving. Now loading the saved datasets...")
    train_ds, val_ds, test_ds, _ = load_processed_datasets(processed_data_dir, batch_size)
    if train_ds is None:
        print("Failed to load datasets after saving. Exiting.")
        exit() # Or handle error appropriately
    # Get input shape from the returned element_spec
    input_shape = element_spec[0].shape
    print(f"Loaded freshly processed datasets. Input shape: {input_shape}")


# In[28]:


# Shuffling, batching, caching, prefetching are now handled within load_processed_datasets
# train_ds = train_ds.shuffle(int(len(train_files)*1.2)) # No longer needed here
# val_ds = val_ds.shuffle(int(len(val_files)*1.2)) # No longer needed here
# test_ds = test_ds.shuffle(int(len(test_files)*1.2)) # No longer needed here


# In[29]:


# Batching, caching, prefetching are now handled within load_processed_datasets
# if not is_batched(train_ds): # No longer needed here
#     train_ds = train_ds.batch(batch_size)
# if not is_batched(val_ds): # No longer needed here
#     val_ds = val_ds.batch(batch_size)
# if not is_batched(test_ds): # No longer needed here
#     test_ds = test_ds.batch(batch_size)

# train_ds = train_ds.cache().prefetch(AUTOTUNE) # No longer needed here
# val_ds = val_ds.cache().prefetch(AUTOTUNE) # No longer needed here


# In[30]:


# Input shape is now determined during dataset loading/processing
# for spectrogram, _ in train_ds.take(1):
#   input_shape = spectrogram.shape[1:]
# print('Input shape:', input_shape) # Already printed above
num_labels = len(label_list)


# In[31]:


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(1,2)),
        layers.BatchNormalization(),
      
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.Conv2D(128, 3, activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(256, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.Conv2D(256, 3, activation='relu'),
        layers.BatchNormalization(),
      
        layers.GlobalMaxPooling2D(),
        layers.Dense(num_labels),
    ])
    model.compile(
        optimizer=tf_keras.optimizers.Adam(),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


# In[32]:


print('Input shape:', input_shape)
model = build_model(input_shape)
model.summary()            


# In[33]:


history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS) 


# In[34]:


date_str = dt.now().strftime("%d%b%Y_%H%M").lower()
print(f"Completed training at {date_str}")


# In[35]:


model_file_name = f"kws_model.h5" 
print(f"Saving model to {model_file_name}")
model.save(model_file_name, overwrite=True)


# In[36]:


## Measure test-set accuracy manually and get values for confusion matrix
test_audio = []
test_labels = []
# Make sure test_ds is unbatched for manual iteration if needed
test_ds_unbatched = test_ds.unbatch() # Create unbatched version if needed

for audio, label in test_ds_unbatched: # Iterate over the unbatched version
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

model_out = model.predict(test_audio)
y_pred = np.argmax(model_out, axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy (manual): {test_acc:.1%}')
# No need to re-batch test_ds here, keep the original batched version for model.evaluate


# In[37]:


## Measure test-set accuracy with the keras built-in function
# Use the original batched test_ds for evaluate
print("Evaluating model on batched test dataset...")
test_loss, test_acc_keras = model.evaluate(test_ds, verbose=2)
print(f'Test set accuracy (Keras): {test_acc_keras:.1%}')


# In[38]:


confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_mtx, xticklabels=label_list, yticklabels=label_list, 
            annot=True, fmt='g')
plt.gca().invert_yaxis() # flip so origin is at bottom left
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


# In[39]:


tpr = np.nan*np.zeros(len(label_list))
fpr = np.nan*np.zeros(len(label_list))
for i in range(4):
    tpr[i]  = confusion_mtx[i,i] / np.sum(confusion_mtx[i,:])
    fpr[i] = (np.sum(confusion_mtx[:,i]) - confusion_mtx[i,i]) / \
      (np.sum(confusion_mtx) - np.sum(confusion_mtx[i,:]))
    print(f"True/False positive rate for '{label_list[i]:9}' = {tpr[i]:.3} / {fpr[i]:.3}")


# In[40]:


info_file_name = model_file_name.split('.')[0] + '.txt'
with open(info_file_name, 'w') as fpo:
    fpo.write(f"i16min            = {i16min           }\n")
    fpo.write(f"i16max            = {i16max           }\n")
    fpo.write(f"fsamp             = {fsamp            }\n")
    fpo.write(f"wave_length_ms    = {wave_length_ms   }\n")
    fpo.write(f"wave_length_samps = {wave_length_samps}\n")
    fpo.write(f"window_size_ms    = {window_size_ms   }\n")
    fpo.write(f"window_step_ms    = {window_step_ms   }\n")
    fpo.write(f"num_filters       = {num_filters      }\n")
    fpo.write(f"use_microfrontend = {use_microfrontend}\n")
    fpo.write(f"label_list        = {label_list}\n")
    # Use the input_shape determined earlier
    fpo.write(f"spectrogram_shape = {input_shape}\n")
    # Use the Keras evaluation accuracy for consistency
    fpo.write(f"Test set accuracy =  {test_acc_keras:.1%}\n")
    for i in range(len(label_list)): # Iterate through all labels
        if i < len(tpr): # Check bounds just in case confusion matrix size differs
          fpo.write(f"tpr_{label_list[i]:9} = {tpr[i]:.3}\n")
          fpo.write(f"fpr_{label_list[i]:9} = {fpr[i]:.3}\n")
        else:
          fpo.write(f"tpr_{label_list[i]:9} = N/A\n")
          fpo.write(f"fpr_{label_list[i]:9} = N/A\n")


# In[41]:


print(f"Wrote description to {info_file_name}")
#get_ipython().system('cat $info_file_name')


# In[42]:


metrics = history.history
plt.subplot(2,1,1)
plt.semilogy(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['training', 'validation'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.subplot(2,1,2)
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['training', 'validation'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
plt.savefig('training_curves.png')


# In[43]:


## Measure test-set accuracy with the keras built-in function
# test_loss, test_acc = model.evaluate(test_ds, verbose=2) # Already done in cell [37]
print(f"Final Test Loss: {test_loss}, Final Test Accuracy: {test_acc_keras}")


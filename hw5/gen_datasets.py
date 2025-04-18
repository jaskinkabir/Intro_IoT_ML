import tensorflow as tf
import tf_keras
import numpy as np
import os
import pathlib
import glob
import random
import pickle
from enum import Enum
from tqdm import tqdm
import time

# Assuming frontend_op is available or can be imported if needed for get_spectrogram
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
    use_microfrontend_op = True
except ImportError:
    print("Warning: audio_microfrontend_op not found. Spectrogram generation might fail.")
    use_microfrontend_op = False


# --- Configuration ---

class DatasetType(Enum):
    MINI = "mini-speech"
    FULL = "full-speech-files"

# --- Default Configuration (can be overridden) ---
SELECTED_DATASET = DatasetType.MINI
COMMANDS = ['stop', 'go']
SILENCE_STR = "_silence"
UNKNOWN_STR = "_unknown"
LIMIT_POSITIVE_SAMPLES = False
MAX_WAVS_PER_COMMAND = {'stop': 50, 'go': 250} # Example limits if LIMIT_POSITIVE_SAMPLES is True
NOISY_REPS_OF_KNOWN = [0.05, 0.1, 0.15, 0.2, 0.25, .1, .1, .1] # Noise levels for augmentation

# Audio processing parameters
FSAMP = 16000 # sampling rate
WAVE_LENGTH_MS = 1000 # 1 sec audio
WAVE_LENGTH_SAMPS = int(WAVE_LENGTH_MS * FSAMP / 1000)
WINDOW_SIZE_MS = 64
WINDOW_STEP_MS = 48
NUM_FILTERS = 32
I16MIN = -2**15
I16MAX = 2**15 - 1

# Paths
HOME_DIR = os.getenv("HOME", ".") # Default to current dir if HOME is not set
BASE_DATA_DIR = os.path.join(HOME_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'processed_speech_commands') # Output directory

AUTOTUNE = tf.data.experimental.AUTOTUNE

# --- Helper Functions ---

def download_dataset(dataset_type: DatasetType, base_data_dir: str):
    """Downloads the specified dataset if it doesn't exist."""
    if dataset_type == DatasetType.MINI:
        data_dir = pathlib.Path(os.path.join(base_data_dir, 'mini_speech_commands'))
        if not data_dir.exists():
            print(f"Downloading {dataset_type.value}...")
            tf_keras.utils.get_file('mini_speech_commands.zip',
                                    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                    extract=True,
                                    #cache_dir=base_data_dir, # Save in base_data_dir
                                    cache_subdir=base_data_dir) # Extract directly into base_data_dir/mini_speech_commands
            print(f"Downloaded and extracted to {data_dir}")
        else:
            print(f"Dataset {dataset_type.value} found at {data_dir}")
        return data_dir
    elif dataset_type == DatasetType.FULL:
        data_dir = pathlib.Path(os.path.join(base_data_dir, 'speech_commands_v0.02'))
        if not data_dir.exists():
             print(f"Downloading {dataset_type.value}...")
             # Note: User might need to download manually or adjust this part
             # For now, we just check existence and raise error.
             # tf_keras.utils.get_file('speech_commands_v0.02.tar.gz',
             #                         origin="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
             #                         extract=True,
             #                         archive_format='tar',
             #                         cache_dir=base_data_dir,
             #                         cache_subdir='')
             # print(f"Downloaded and extracted to {data_dir}")
             raise RuntimeError(f"Full speech commands dataset not found at {data_dir}. Please download it manually (e.g., wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) and extract it there.")
        else:
            print(f"Dataset {dataset_type.value} found at {data_dir}")
        return data_dir
    else:
        raise ValueError("Invalid dataset type specified.")

def get_file_lists(data_dir: pathlib.Path, dataset_type: DatasetType):
    """Gets lists of training, validation, and test files."""
    print(f"Generating file lists for {dataset_type.value}...")
    all_files = glob.glob(os.path.join(str(data_dir), '*', '*.wav'))
    random.shuffle(all_files) # Shuffle once initially
    num_samples = len(all_files)
    print(f'Total samples found: {num_samples}')

    if dataset_type == DatasetType.MINI:
        num_train_files = int(0.8 * num_samples)
        num_val_files = int(0.1 * num_samples)
        train_files = all_files[:num_train_files]
        val_files = all_files[num_train_files: num_train_files + num_val_files]
        test_files = all_files[num_train_files + num_val_files:]
    elif dataset_type == DatasetType.FULL:
        # Use validation_list.txt and testing_list.txt for full dataset
        fname_val_files = os.path.join(data_dir, 'validation_list.txt')
        with open(fname_val_files) as fpi_val:
            val_files_rel = fpi_val.read().splitlines()
        val_files = [os.path.join(data_dir, fn.replace('/', os.sep)) for fn in val_files_rel]

        fname_test_files = os.path.join(data_dir, 'testing_list.txt')
        with open(fname_test_files) as fpi_tst:
            test_files_rel = fpi_tst.read().splitlines()
        test_files = [os.path.join(data_dir, fn.replace('/', os.sep)) for fn in test_files_rel]

        # Train with everything else, excluding background noise and val/test files
        all_files_set = set(all_files)
        # Exclude files in directories starting with '_' (like _background_noise_)
        train_files = [f for f in all_files if f.split(os.sep)[-2][0] != '_']
        train_files = list(set(train_files) - set(val_files) - set(test_files))
    else:
        raise ValueError("Invalid dataset type specified.")

    print(f'Training set size: {len(train_files)}')
    print(f'Validation set size: {len(val_files)}')
    print(f'Test set size: {len(test_files)}')
    return train_files, val_files, test_files

def limit_samples(train_files: list, commands: list, max_samples_per_command: dict):
    """Limits the number of training samples for specified commands."""
    print("Limiting samples for specified commands...")
    counts = {cmd: 0 for cmd in commands}
    limited_train_files = []
    # Shuffle before limiting to get a random subset
    random.shuffle(train_files)

    for f in train_files:
        label = f.split(os.sep)[-2]
        if label in commands:
            if counts[label] < max_samples_per_command.get(label, float('inf')):
                limited_train_files.append(f)
                counts[label] += 1
        else:
            # Keep all non-target command files (or apply other limits if needed)
            limited_train_files.append(f)

    print(f"Limited training set size: {len(limited_train_files)}")
    print(f"Counts per limited command: {counts}")
    return limited_train_files

# --- Audio Processing Functions (Moved/Adapted from speech_training.py) ---

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path, label_list, unknown_str):
    parts = tf.strings.split(file_path, os.path.sep)
    # Check if the directory name (parts[-2]) is in our target label_list
    in_set = tf.reduce_any(parts[-2] == label_list)
    # If it is, use it as the label; otherwise, use unknown_str
    label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(unknown_str))
    return label

def get_waveform_and_label(file_path, label_list, unknown_str):
    label = get_label(file_path, label_list, unknown_str)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform, wave_length_samps, i16min, i16max, fsamp, num_filters, window_size_ms, window_step_ms):
    if not use_microfrontend_op:
        raise RuntimeError("audio_microfrontend_op is required for get_spectrogram.")
    # Pad or truncate waveform to wave_length_samps
    padding = tf.maximum(0, wave_length_samps - tf.shape(waveform)[0])
    waveform = tf.pad(waveform, [[0, padding]])
    waveform = waveform[:wave_length_samps] # Ensure exact length

    # Cast to int16
    waveform_int16 = tf.cast(waveform * (i16max - i16min) / 2.0, tf.int16)

    spectrogram = frontend_op.audio_microfrontend(
        waveform_int16,
        sample_rate=fsamp,
        num_channels=num_filters,
        window_size=window_size_ms,
        window_step=window_step_ms
    )
    return spectrogram

def create_silence_dataset(num_waves, samples_per_wave, rms_noise_range=[0.01, 0.2], silent_label="_silence"):
    rng = np.random.default_rng()
    rms_noise_levels = rng.uniform(low=rms_noise_range[0], high=rms_noise_range[1], size=num_waves)
    rand_waves = np.zeros((num_waves, samples_per_wave), dtype=np.float32)
    for i in range(num_waves):
        rand_waves[i, :] = rms_noise_levels[i] * rng.standard_normal(samples_per_wave)
    labels = [silent_label] * num_waves
    return tf.data.Dataset.from_tensor_slices((rand_waves, labels))

def copy_with_noise(ds_input, wave_length_samps, rms_level=0.25):
    rng = tf.random.Generator.from_seed(1234)
    wave_shape = tf.constant((wave_length_samps,))
    def add_noise(waveform, label):
        noise = rms_level * rng.normal(shape=wave_shape)
        # Ensure waveform is padded/truncated before adding noise
        padding = tf.maximum(0, wave_length_samps - tf.shape(waveform)[0])
        waveform = tf.pad(waveform, [[0, padding]])
        waveform = waveform[:wave_length_samps]
        noisy_wave = waveform + noise
        return noisy_wave, label
    return ds_input.map(add_noise, num_parallel_calls=AUTOTUNE)

# --- Spectrogram Conversion ---

def preprocess_to_spectrograms(files: list, commands: list, label_list: list, silence_str: str, unknown_str: str,
                               wave_length_samps: int, fsamp: int, num_filters: int, window_size_ms: int, window_step_ms: int,
                               i16min: int, i16max: int,
                               num_silent=None, noisy_reps_of_known=None, set_name=""):
    """Converts audio files to a spectrogram dataset."""
    if num_silent is None:
        num_silent = int(0.2 * len(files)) + 1
    print(f"\n--- Processing {set_name} Set ({len(files)} files) ---")

    # Create dataset from file paths
    files_ds = tf.data.Dataset.from_tensor_slices(files)

    # Define the mapping function locally to capture variables from the outer scope
    # This might help tf.function/AutoGraph handle types correctly.
    def _local_get_waveform_and_label(file_path):
        # Explicitly cast file_path to string just in case
        file_path_str = tf.cast(file_path, tf.string)
        # Re-define get_label logic locally or ensure it correctly uses captured vars
        # Using local logic for clarity:
        parts = tf.strings.split(file_path_str, os.path.sep)
        # Check if the directory name (parts[-2]) is in our target label_list
        in_set = tf.reduce_any(parts[-2] == label_list)
        # If it is, use it as the label; otherwise, use unknown_str
        label = tf.cond(in_set, lambda: parts[-2], lambda: tf.constant(unknown_str))

        audio_binary = tf.io.read_file(file_path_str) # Use the casted string path
        waveform = decode_audio(audio_binary)
        return waveform, label

    # Map to waveforms and labels using the locally defined function
    waveform_ds = files_ds.map(_local_get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    # Augmentation: Add noisy copies of target commands
    num_added_noisy = 0
    if noisy_reps_of_known is not None and len(noisy_reps_of_known) > 0:
        print(f"Adding noisy copies for commands: {commands} with levels: {noisy_reps_of_known}")
        # Filter for target commands only
        ds_only_cmds = waveform_ds.filter(lambda w, l: tf.reduce_any(l == commands))
        # Cache the filtered dataset for efficiency if it's reused multiple times
        ds_only_cmds = ds_only_cmds.cache() # Cache before concatenating multiple times

        # ... (optional count estimation code removed for brevity) ...

        original_waveform_ds = waveform_ds # Keep the original

        # Initialize augmented_ds with the correct element spec
        # Use an empty generator and specify the output signature based on ds_only_cmds
        element_spec = ds_only_cmds.element_spec
        augmented_ds = tf.data.Dataset.from_generator(
            lambda: (i for i in []), # Empty generator
            output_signature=element_spec
        )

        for noise_level in noisy_reps_of_known:
            noisy_copy_ds = copy_with_noise(ds_only_cmds, wave_length_samps, rms_level=noise_level)
            augmented_ds = augmented_ds.concatenate(noisy_copy_ds)
            # ... (optional count update code removed for brevity) ...

        # Concatenate original and augmented datasets
        waveform_ds = original_waveform_ds.concatenate(augmented_ds)
        # ... (reporting augmentation happened) ...

    # Add silence
    if num_silent > 0:
        print(f"Adding {num_silent} silence samples.")
        silent_wave_ds = create_silence_dataset(num_silent, wave_length_samps, silent_label=silence_str)
        waveform_ds = waveform_ds.concatenate(silent_wave_ds)

    # --- Convert waveforms to spectrograms and labels to IDs ---
    print("Converting waveforms to spectrograms...")

    # Get one sample spectrogram to determine shape
    temp_wav, temp_label = next(iter(waveform_ds))
    one_spec = get_spectrogram(temp_wav, wave_length_samps, i16min, i16max, fsamp, num_filters, window_size_ms, window_step_ms)
    spec_shape_no_batch = one_spec.shape + (1,) # Add channel dimension

    # Estimate total number of waves after augmentation/silence
    # This can be slow or inaccurate for complex datasets
    # total_waves = tf.data.experimental.cardinality(waveform_ds)
    # if total_waves == tf.data.experimental.UNKNOWN_CARDINALITY:
    #     print("Warning: Unknown dataset size, iterating to count...")
    #     total_waves = 0
    #     for _ in waveform_ds: total_waves += 1
    # print(f"Total waves to process: {total_waves}")

    # Pre-allocate numpy arrays (if memory allows and size is known)
    # This is often faster than appending but requires knowing the size
    # spec_grams = np.zeros((total_waves,) + spec_shape_no_batch, dtype=np.float32)
    # labels = np.zeros(total_waves, dtype=np.int64)
    # idx = 0

    # Alternative: Use map (potentially slower if ops aren't vectorized well, but more memory efficient)
    def waveform_to_spec_and_label_id(waveform, label):
        spectrogram = get_spectrogram(waveform, wave_length_samps, i16min, i16max, fsamp, num_filters, window_size_ms, window_step_ms)
        spectrogram = tf.expand_dims(spectrogram, axis=-1) # Add channel dim
        # Convert string label to integer ID
        label_id = tf.argmax(tf.cast(label == label_list, tf.int64))
        return spectrogram, label_id

    # Use map for the conversion
    output_ds = waveform_ds.map(waveform_to_spec_and_label_id, num_parallel_calls=AUTOTUNE)

    # Optional: Add progress bar for map operation (requires tf-nightly or custom callback)
    # For simplicity, we'll rely on tqdm for the initial file processing if needed elsewhere.

    print(f"Finished converting {set_name} set to spectrograms.")
    return output_ds


# --- Save/Load Functions ---

def save_datasets(train_ds, val_ds, test_ds, element_spec, save_dir):
    """Saves the processed datasets and element spec."""
    print(f"\nSaving datasets to {save_dir}...")
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_save_path = os.path.join(save_dir, 'train')
    val_save_path = os.path.join(save_dir, 'val')
    test_save_path = os.path.join(save_dir, 'test')
    spec_save_path = os.path.join(save_dir, 'element_spec.pkl')

    print(f"Saving training dataset to {train_save_path}...")
    tf.data.Dataset.save(train_ds, train_save_path)
    print(f"Saving validation dataset to {val_save_path}...")
    tf.data.Dataset.save(val_ds, val_save_path)
    print(f"Saving test dataset to {test_save_path}...")
    tf.data.Dataset.save(test_ds, test_save_path)

    print(f"Saving element spec to {spec_save_path}...")
    with open(spec_save_path, 'wb') as f:
        pickle.dump(element_spec, f)

    print("Datasets and spec saved successfully.")

def load_datasets(save_dir, batch_size=None):
    """Loads preprocessed datasets from disk."""
    print(f"\nLoading processed datasets from {save_dir}...")
    t_start = time.time()

    train_save_path = os.path.join(save_dir, 'train')
    val_save_path = os.path.join(save_dir, 'val')
    test_save_path = os.path.join(save_dir, 'test')
    spec_save_path = os.path.join(save_dir, 'element_spec.pkl')

    if not all(os.path.exists(p) for p in [train_save_path, val_save_path, test_save_path, spec_save_path]):
        print("Error: Not all required dataset files/spec file found.")
        return None, None, None, None

    # Load element spec separately (for return value, not for loading)
    loaded_spec = None
    try:
        with open(spec_save_path, 'rb') as f:
            loaded_spec = pickle.load(f)
        print("Loaded element spec from pickle.")
    except Exception as e:
        print(f"Warning: Error loading element spec from pickle: {e}")
        # Continue loading datasets, TF might infer the spec

    # Load datasets without explicitly passing the pickled spec
    try:
        print("Loading training dataset...")
        # Let TF infer the spec from saved data
        train_ds = tf.data.Dataset.load(train_save_path)
        print("Loading validation dataset...")
        val_ds = tf.data.Dataset.load(val_save_path)
        print("Loading test dataset...")
        test_ds = tf.data.Dataset.load(test_save_path)
        print("Datasets loaded successfully.")
        # Verify the inferred spec if the pickled one was loaded
        if loaded_spec and train_ds.element_spec != loaded_spec:
             print("Warning: Inferred element spec differs from pickled spec.")
             print(f"  Inferred: {train_ds.element_spec}")
             print(f"  Pickled:  {loaded_spec}")
             # Use the inferred spec from the loaded dataset as it's more likely correct
             loaded_spec = train_ds.element_spec

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None

    # Apply post-loading transformations (optional batching, shuffling, prefetching)
    # Shuffling should generally be done before batching
    train_ds = train_ds.shuffle(10000) # Adjust buffer size as needed
    val_ds = val_ds.shuffle(1000)   # Adjust buffer size as needed

    if batch_size:
        print(f"Batching datasets with batch size: {batch_size}")
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size) # Batch test set for efficient evaluation

    # Prefetching is usually the last step
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE) # Prefetch test set

    t_end = time.time()
    print(f"Dataset loading and preparation finished in {t_end - t_start:.2f} seconds.")

    # Return the loaded datasets and the (potentially updated) spec
    return train_ds, val_ds, test_ds, loaded_spec


# --- Main Execution ---

def gen_datasets_and_save(
    dataset_type: DatasetType = SELECTED_DATASET,
    commands: list = COMMANDS,
    silence_str: str = SILENCE_STR,
    unknown_str: str = UNKNOWN_STR,
    limit_positive_samples: bool = LIMIT_POSITIVE_SAMPLES,
    max_wavs_per_command: dict = MAX_WAVS_PER_COMMAND,
    # num_silent: int = None, # num_silent calculation depends on file list size, handle inside
    noisy_reps_of_known: list = NOISY_REPS_OF_KNOWN,
    wave_length_samps: int = WAVE_LENGTH_SAMPS,
    fsamp: int = FSAMP,
    num_filters: int = NUM_FILTERS,
    window_size_ms: int = WINDOW_SIZE_MS,
    window_step_ms: int = WINDOW_STEP_MS,
    i16min: int = I16MIN,
    i16max: int = I16MAX,
    base_data_dir: str = BASE_DATA_DIR,
    processed_data_dir: str = PROCESSED_DATA_DIR,
    save_datasets_to_disk: bool = True, # Add option to save
):
    """
    Generates and optionally saves spectrogram datasets.

    Args:
        dataset_type: Type of dataset (MINI or FULL).
        commands: List of target command words.
        silence_str: Label for silence.
        unknown_str: Label for unknown words.
        limit_positive_samples: Whether to limit samples for commands.
        max_wavs_per_command: Dictionary mapping commands to max samples.
        noisy_reps_of_known: List of noise levels for augmentation.
        wave_length_samps: Length of audio clips in samples.
        fsamp: Sampling frequency.
        num_filters: Number of Mel filters.
        window_size_ms: Spectrogram window size in ms.
        window_step_ms: Spectrogram window step in ms.
        i16min: Minimum int16 value.
        i16max: Maximum int16 value.
        base_data_dir: Directory containing raw datasets.
        processed_data_dir: Directory to save/load processed datasets.
        save_datasets_to_disk: If True, save the generated datasets.

    Returns:
        Tuple: (train_ds, val_ds, test_ds, element_spec)
    """
    print("Starting dataset generation process...")

    # 1. Configuration (Use function parameters)
    label_list = commands.copy()
    label_list.insert(0, silence_str)
    label_list.insert(1, unknown_str)
    print(f"Target commands: {commands}")
    print(f"Full label list: {label_list}")

    # 2. Download Data
    raw_data_dir = download_dataset(dataset_type, base_data_dir)

    # 3. Get File Lists
    train_files, val_files, test_files = get_file_lists(raw_data_dir, dataset_type)

    # 4. Limit Samples (Optional)
    if limit_positive_samples:
        train_files = limit_samples(train_files, commands, max_wavs_per_command)

    # Calculate num_silent based on potentially limited train_files
    num_silent_train = int(0.2 * len(train_files)) + 1
    num_silent_val = int(0.2 * len(val_files)) + 1
    num_silent_test = int(0.2 * len(test_files)) + 1


    # 5. Preprocess to Spectrograms
    # Note: Pass all necessary parameters explicitly using function args
    train_ds = preprocess_to_spectrograms(
        train_files, commands, label_list, silence_str, unknown_str,
        wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms,
        i16min, i16max, num_silent=num_silent_train,
        noisy_reps_of_known=noisy_reps_of_known, set_name="Training"
    )
    val_ds = preprocess_to_spectrograms(
        val_files, commands, label_list, silence_str, unknown_str,
        wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms,
        i16min, i16max, num_silent=num_silent_val, set_name="Validation"
    )
    test_ds = preprocess_to_spectrograms(
        test_files, commands, label_list, silence_str, unknown_str,
        wave_length_samps, fsamp, num_filters, window_size_ms, window_step_ms,
        i16min, i16max, num_silent=num_silent_test, set_name="Test"
    )

    # 6. Get Element Spec (from the processed training dataset)
    element_spec = train_ds.element_spec
    print(f"\nDetermined element spec: {element_spec}")

    # 7. Save Datasets (Optional)
    if save_datasets_to_disk:
        save_datasets(train_ds, val_ds, test_ds, element_spec, processed_data_dir)
    else:
        print("\nSkipping saving datasets to disk.")

    print("\nDataset generation complete.")
    return train_ds, val_ds, test_ds, element_spec


if __name__ == "__main__":
    # This block allows running the script standalone to generate and save datasets
    # using the default configurations defined at the top of the file.
    print("Running gen_datasets.py as standalone script...")

    # Call the main function with default parameters (which are the globals)
    gen_datasets_and_save(
        dataset_type=SELECTED_DATASET,
        commands=COMMANDS,
        silence_str=SILENCE_STR,
        unknown_str=UNKNOWN_STR,
        limit_positive_samples=LIMIT_POSITIVE_SAMPLES,
        max_wavs_per_command=MAX_WAVS_PER_COMMAND,
        noisy_reps_of_known=NOISY_REPS_OF_KNOWN,
        wave_length_samps=WAVE_LENGTH_SAMPS,
        fsamp=FSAMP,
        num_filters=NUM_FILTERS,
        window_size_ms=WINDOW_SIZE_MS,
        window_step_ms=WINDOW_STEP_MS,
        i16min=I16MIN,
        i16max=I16MAX,
        base_data_dir=BASE_DATA_DIR,
        processed_data_dir=PROCESSED_DATA_DIR,
        save_datasets_to_disk=True # Ensure saving when run standalone
    )

    # Optional: Load and verify after saving
    print("\nVerifying saved datasets by reloading...")
    # Use batch_size=32 for verification loading example
    loaded_train_ds, loaded_val_ds, loaded_test_ds, loaded_spec = load_datasets(PROCESSED_DATA_DIR, batch_size=32)

    if loaded_train_ds:
        print("Verification successful. Example batch spec:")
        print(loaded_train_ds.element_spec)
        # You could take(1) and print shapes/values here too
        for spec, label in loaded_train_ds.take(1):
            print("Example train batch shapes - Spectrogram:", spec.shape, "Labels:", label.shape)
    else:
        print("Verification failed.")






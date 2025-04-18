import tf_keras
import tensorflow as tf
from tf_keras import models, layers, metrics
import os
from gen_datasets_new import gen_datasets_and_save, load_datasets, DatasetType
from classifier import Classifier
import numpy as np # Import numpy
import json

model_name = 'best_mini'

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

home_dir = os.getenv("HOME")
raw_dataset_path = os.path.join(home_dir, "iot/datasets/mini_speech_commands")
processed_dataset_path = os.path.join(home_dir, "iot/datasets/processed_mini_speech_commands")

# Rename 'spec' to 'loaded_spec_info' to clarify it's just informational now
# Also get label list if possible (assuming load_datasets or gen_datasets provides it)
# For now, define it manually based on train_model.py/gen_datasets_new.py defaults
# TODO: Ideally, load this from a saved file or dataset metadata
commands = ['stop', 'go']
silence_str = "_silence"
unknown_str = "_unknown"
label_list = [silence_str, unknown_str] + commands # Should be ['_silence', '_unknown', 'stop', 'go']
num_classes = len(label_list)

train, val, test, loaded_spec_info = load_datasets(processed_dataset_path, batch_size)

model = tf_keras.models.load_model('training/best_model.keras')

# Compile with only accuracy, as other metrics cause shape issues here
model.compile(
    optimizer=tf_keras.optimizers.Adam(learning_rate=0.001),
    loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy', # Use 'accuracy' or tf.keras.metrics.SparseCategoricalAccuracy()
    ]
)

# Evaluate basic loss and accuracy
print("Evaluating model loss and accuracy...")
train_hist = model.evaluate(
    train,
    verbose=1,
    return_dict=True,
)

hist = model.evaluate(
    test,
    verbose=1,
    return_dict=True,
)
print(f"Train Accuracy: {train_hist['accuracy']}")
print(f"Test Accuracy: {hist['accuracy']}")

# --- Manual Calculation of Confusion Matrix and Metrics ---
print("\nCalculating confusion matrix and detailed metrics...")

# 1. Get Predictions
# Need to unbatch the test dataset to predict sample by sample or predict on batches and concatenate
all_predictions = []
all_labels = []
print("Gathering predictions and labels from test set...")
for features, labels in test: # Iterate through batches
    batch_predictions = model.predict_on_batch(features)
    all_predictions.append(batch_predictions)
    all_labels.append(labels.numpy()) # Store labels

# Concatenate results from all batches
predictions_logits = np.concatenate(all_predictions, axis=0)
true_labels = np.concatenate(all_labels, axis=0)

# 2. Convert Logits to Predicted Class Indices
predicted_labels = np.argmax(predictions_logits, axis=1)

# 3. Calculate Confusion Matrix
confusion_mtx = tf.math.confusion_matrix(
    labels=true_labels,
    predictions=predicted_labels,
    num_classes=num_classes # Ensure the matrix has the correct size
).numpy()

print("\nConfusion Matrix:")
print(confusion_mtx)
print("\nMetrics per class:")

# 4. Calculate TP, FP, TN, FN per class
for i in range(num_classes):
    tp = confusion_mtx[i, i]
    fp = np.sum(confusion_mtx[:, i]) - tp
    fn = np.sum(confusion_mtx[i, :]) - tp
    tn = np.sum(confusion_mtx) - (tp + fp + fn)

    # Calculate rates (handle division by zero)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall, Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # Fall-out
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # Miss rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"\nClass: {label_list[i]} ({i})")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    print(f"  TPR (Recall): {tpr:.4f}")
    print(f"  TNR (Specificity): {tnr:.4f}")
    print(f"  FPR: {fpr:.4f}")
    print(f"  FNR: {fnr:.4f}")
    print(f"  Precision: {precision:.4f}")

    # Convert numpy types to standard Python types for JSON serialization
    results = {
        'model_name' : model_name,
        'train_accuracy' : float(train_hist['accuracy']), # Convert to float
        'test_accuracy' : float(hist['accuracy']),       # Convert to float
        'metrics' : {
            'per_class_metrics' : {
                label_list[i]: {
                    'TP': int(tp),         # Convert to int
                    'FP': int(fp),         # Convert to int
                    'FN': int(fn),         # Convert to int
                    'TN': int(tn),         # Convert to int
                    'TPR': float(tpr),     # Convert to float
                    'TNR': float(tnr),     # Convert to float
                    'FPR': float(fpr),     # Convert to float
                    'FNR': float(fnr),     # Convert to float
                    'Precision': float(precision) # Convert to float
                } for i, (tp, fp, fn, tn, tpr, tnr, fpr, fnr, precision) in enumerate(
                    # Recalculate inside comprehension to get original numpy types before casting
                    (
                        (
                            confusion_mtx[i, i],
                            np.sum(confusion_mtx[:, i]) - confusion_mtx[i, i],
                            np.sum(confusion_mtx[i, :]) - confusion_mtx[i, i],
                            np.sum(confusion_mtx) - (np.sum(confusion_mtx[:, i]) + np.sum(confusion_mtx[i, :]) - confusion_mtx[i, i]),
                            (confusion_mtx[i, i] / (np.sum(confusion_mtx[i, :])) if (np.sum(confusion_mtx[i, :])) > 0 else 0.0), # TPR
                            ((np.sum(confusion_mtx) - (np.sum(confusion_mtx[:, i]) + np.sum(confusion_mtx[i, :]) - confusion_mtx[i, i])) / (np.sum(confusion_mtx) - np.sum(confusion_mtx[i, :])) if (np.sum(confusion_mtx) - np.sum(confusion_mtx[i, :])) > 0 else 0.0), # TNR
                            ((np.sum(confusion_mtx[:, i]) - confusion_mtx[i, i]) / (np.sum(confusion_mtx) - np.sum(confusion_mtx[i, :])) if (np.sum(confusion_mtx) - np.sum(confusion_mtx[i, :])) > 0 else 0.0), # FPR
                            ((np.sum(confusion_mtx[i, :]) - confusion_mtx[i, i]) / (np.sum(confusion_mtx[i, :])) if (np.sum(confusion_mtx[i, :])) > 0 else 0.0), # FNR
                            (confusion_mtx[i, i] / (np.sum(confusion_mtx[:, i])) if (np.sum(confusion_mtx[:, i])) > 0 else 0.0)  # Precision
                        ) for i in range(num_classes)
                    )
                )
            }
        },
        # Add confusion matrix (convert to list of lists)
        'confusion_matrix': confusion_mtx.tolist()
    }

    # Ensure the directory exists
    os.makedirs('training', exist_ok=True)

    # Save results to JSON
    output_path = f'training/{model_name}_evaluation_metrics.json'
    print(f"\nSaving evaluation results to {output_path}")
    with open(output_path, 'w+') as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete.")

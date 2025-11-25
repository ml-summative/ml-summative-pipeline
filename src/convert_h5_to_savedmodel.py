"""
convert_h5_to_savedmodel.py
Utility to convert an incompatible Keras HDF5 (`.h5`) or `.keras` model into a TensorFlow SavedModel
This attempts multiple strategies (full load, then rebuild+load_weights with by_name/skip_mismatch)

Usage (PowerShell):
    cd <project_root>\src
    python .\convert_h5_to_savedmodel.py ../models/traffic_net_model.keras ../models/traffic_net_model_saved

If conversion succeeds, point your prediction service at the SavedModel directory.
"""

import sys
from pathlib import Path
import pickle
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def build_model(img_size=(224,224), num_classes=4):
    input_shape = (*img_size, 3)
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main(h5_path, out_dir):
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)

    if not h5_path.exists():
        print(f"HDF5/.keras model not found: {h5_path}")
        return 1

    # Try standard load_model
    print("Trying keras.models.load_model(...) on the file...")
    try:
        m = keras.models.load_model(h5_path)
        print("Successfully loaded full model. Saving as SavedModel...")
        m.save(out_dir)
        print(f"SavedModel written to: {out_dir}")
        return 0
    except Exception:
        print("Full model load failed:")
        traceback.print_exc()

    # Try to read class indices to determine num_classes
    class_idx_path = h5_path.parent / 'class_indices.pkl'
    if class_idx_path.exists():
        try:
            with open(class_idx_path, 'rb') as f:
                class_indices = pickle.load(f)
            inv = {v:k for k,v in class_indices.items()}
            class_names = [inv[i] for i in range(len(inv))]
            print(f"Loaded class names from {class_idx_path}: {class_names}")
        except Exception:
            print("Failed to load class_indices.pkl; falling back to 4 classes.")
            class_names = ['sparse_traffic','dense_traffic','accident','fire']
    else:
        print("No class_indices.pkl found; defaulting to 4 classes.")
        class_names = ['sparse_traffic','dense_traffic','accident','fire']

    num_classes = len(class_names)
    print(f"Rebuilding architecture with num_classes={num_classes} and img_size=(224,224)")
    model = build_model(img_size=(224,224), num_classes=num_classes)

    # Attempt to load weights by name and skip mismatches
    print("Attempting model.load_weights(..., by_name=True, skip_mismatch=True) to salvage compatible weights...")
    try:
        model.load_weights(h5_path, by_name=True, skip_mismatch=True)
        print("Weights loaded (by_name, skip_mismatch=True). Saving as SavedModel...")
        # Compile minimally for saving
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.save(out_dir)
        print(f"SavedModel written to: {out_dir}")
        return 0
    except Exception:
        print("Failed to load weights into rebuilt model:")
        traceback.print_exc()
        print("Conversion unsuccessful. Recommended: re-save the model from the training env using `model.save(<dir>)`.")
        return 2


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python convert_h5_to_savedmodel.py <path/to/model.h5|.keras> <out/saved_model_dir>")
        sys.exit(1)

    h5_path = sys.argv[1]
    out_dir = sys.argv[2]
    rc = main(h5_path, out_dir)
    sys.exit(rc)
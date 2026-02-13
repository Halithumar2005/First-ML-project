import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

# Reduce noisy warnings from TensorFlow / Keras / numpy
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Ensure TensorFlow logger is quiet
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load pretrained CNN once
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

CLASSES = ["electric bus", "electric car"]

def extract_features(folder):
    features, labels = [], []

    for label, cls in enumerate(CLASSES):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            continue

        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            if not os.path.isfile(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # OpenCV loads images in BGR; convert to RGB expected by preprocess_input
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                continue

            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            try:
                feature = feature_extractor.predict(img, verbose=0)
            except Exception:
                # Skip problematic images rather than crashing
                continue

            features.append(feature.flatten())
            labels.append(label)

    return np.array(features), np.array(labels)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def build_encoder():
    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling=None)
    base_model.trainable = False
    return base_model

def preprocess_image(image_path: str, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(encoder, image_paths: list, batch_size: int = 32):
    features = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = np.array([preprocess_image(p) for p in batch_paths])
        batch_features = encoder.predict(batch_images, verbose=0)
        batch_features = batch_features.reshape(batch_features.shape[0], -1, batch_features.shape[-1])

        for path, feat in zip(batch_paths, batch_features):
            features[path] = feat

    return features
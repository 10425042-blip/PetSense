"""
Script chuyen doi model .h5 sang .tflite
Chi can chay 1 lan tren may local (noi da cai TensorFlow)
"""
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_PATH = os.path.join(BASE_DIR, 'pet_breed_model.h5')
TFLITE_PATH = os.path.join(BASE_DIR, 'pet_breed_model.tflite')

print("Dang load model .h5 ...")
model = tf.keras.models.load_model(H5_PATH)
print(f"Model loaded: input shape = {model.input_shape}")

print("Dang convert sang TFLite ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimize de giam kich thuoc file
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"Done! File: {TFLITE_PATH}")
print(f"Kich thuoc: {size_mb:.2f} MB")

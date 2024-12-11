import tensorflow as tf
import os

# Correct path to your Keras model
keras_model_path = "/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_9_Homework"  # adapt path as necessary. Download model as per homework instructions

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
tflite_model_path = "model_2024_hairstyle.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

# Get the size of the converted model
tflite_model_size = os.path.getsize(tflite_model_path)
print(f"Converted TFLite model size: {tflite_model_size / 1024:.2f} KB")

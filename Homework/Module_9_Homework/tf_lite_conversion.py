import tensorflow as tf
import os

# Code for Question 1 (converting model to tf lite)

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


# Code for Question 2 (Output index)
import tensorflow as tf

# Load the TFLite model
tflite_model_path = "/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_9_Homework/model_2024_hairstyle.tflite"

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# Allocate tensors (necessary to initialize the model)
interpreter.allocate_tensors()

# Get the details of the model's input and output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the output index and details
print(f"Output index: {output_details[0]['index']}")
print(f"Output name: {output_details[0]['name']}")
print(f"Output shape: {output_details[0]['shape']}")

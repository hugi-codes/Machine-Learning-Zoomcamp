# This script serves to answer Question 3 and Question 4

import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import tensorflow as tf

# Function to download and prepare the image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Load the Keras model to get the input size
keras_model_path = "/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_9_Homework/model_2024_hairstyle.tflite"
model = tf.keras.models.load_model(keras_model_path)
input_shape = model.input_shape
target_size = (input_shape[1], input_shape[2])  # (height, width)

# Download and prepare the image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img, target_size)

# Convert the image to a numpy array and normalize
img_array = np.array(img) / 255.0  # Normalizing pixel values to [0, 1]

# Get the R channel value of the first pixel
r_channel_value = img_array[0, 0, 0]  # First pixel, R channel (Red)
print(f"R channel value of the first pixel: {r_channel_value}")


# Question 4
interpreter = tf.lite.Interpreter(model_path=keras_model_path)
interpreter.allocate_tensors()

img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

# Run inference
interpreter.invoke()

# Get the output tensor
output = interpreter.get_tensor(output_details[0]['index'])

# Print the output of the model
print("Model Output:", output)
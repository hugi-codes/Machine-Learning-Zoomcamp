import os
import pickle
from flask import Flask, request, jsonify
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the only `.pkl` file in the directory
pkl_files = [file for file in os.listdir(script_dir) if file.endswith('.pkl')]

# Ensure there is exactly one `.pkl` file
if len(pkl_files) != 1:
    raise ValueError(f"Expected exactly one .pkl file in the directory, but found {len(pkl_files)}: {pkl_files}")

# Load the model dynamically
pickle_filepath = os.path.join(script_dir, pkl_files[0])
with open(pickle_filepath, 'rb') as file:
    best_model = pickle.load(file)

# Print a success message after loading the model
print(f"The model has been loaded successfully from '{pickle_filepath}'.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict using the loaded model.
    Accepts JSON input with features and returns predictions.
    """
    # Get input data from the request
    input_data = request.json
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Convert the input data to a numpy array for prediction
        X_new = np.array([list(input_data.values())])

        # Generate prediction
        prediction = best_model.predict(X_new)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9696)

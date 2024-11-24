import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler

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


EXPECTED_FEATURES = [
    'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
    'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'FastingBS_1', 'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up'
]

def preprocess_input(data):
    """
    Preprocess incoming JSON data to align with the model's expected features.
    Includes:
    - Categorical transformations
    - One-hot encoding
    - Min-Max scaling
    - Reindexing to ensure feature alignment
    """
    # Convert incoming data to a DataFrame
    df = pd.DataFrame([data])

    # Define preprocessing parameters
    categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

    # Ensure proper column types for categorical data
    for col in categorical_columns:
        if col in df:
            df[col] = df[col].astype('category')

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Min-Max Scaling for numerical columns
    scaler = MinMaxScaler()
    if any(col in df for col in numerical_columns):
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Reindex to ensure all expected features are present
    df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    return df


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict using the loaded model.
    Accepts JSON input with features, preprocesses it, and returns predictions.
    """
    # Get input data from the request
    input_data = request.json
    if not input_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Ensure the preprocessed data matches the model's expected features
        prediction_proba = best_model.predict_proba(preprocessed_data)[:, 1][0].item()  # Convert to native Python float

        # Determine heart disease risk
        result = "Patient has Heart Disease" if prediction_proba >= 0.5 else "Patient has no Heart Disease"

        # Return the prediction and probability
        return jsonify({
            'prediction_proba': prediction_proba,  # Native float
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=9696)

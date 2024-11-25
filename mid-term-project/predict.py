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

# Load all .pkl files in the script's directory
pkl_files = {file: os.path.join(script_dir, file) for file in os.listdir(script_dir) if file.endswith('.pkl')}

# Load the DictVectorizer
dict_vectorizer_file = next((f for f in pkl_files if "Dict_Vectorizer.pkl" in f), None)
if not dict_vectorizer_file:
    raise FileNotFoundError("Dict_Vectorizer.pkl not found.")
with open(pkl_files[dict_vectorizer_file], 'rb') as f:
    dict_vectorizer = pickle.load(f)

# Load the MinMaxScaler
minmax_scaler_file = next((f for f in pkl_files if "MinMax_scaler.pkl" in f), None)
if not minmax_scaler_file:
    raise FileNotFoundError("MinMax_scaler.pkl not found.")
with open(pkl_files[minmax_scaler_file], 'rb') as f:
    minmax_scaler = pickle.load(f)

# Load the model (any file containing 'model' in its filename)
model_file = next((f for f in pkl_files if "Model" in f), None)
if not model_file:
    raise FileNotFoundError("Model pickle file not found.")
with open(pkl_files[model_file], 'rb') as f:
    best_model = pickle.load(f)

# Print success messages
print(f"DictVectorizer loaded successfully from '{dict_vectorizer_file}'.")
print(f"MinMaxScaler loaded successfully from '{minmax_scaler_file}'.")
print(f"Model loaded successfully from '{model_file}'.")

def preprocess_input(data):
    """
    Preprocess incoming JSON data to align with the model's expectations.
    Includes:
    - Categorical transformations (One-Hot Encoding)
    - Min-Max scaling
    - Reindexing to ensure alignment with features used during model training
    """
    # Convert incoming data to a DataFrame
    df = pd.DataFrame([data])

    # Define the categorical columns
    categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Ensure the categorical columns are of category type
    for col in categorical_columns:
        if col in df:
            df[col] = df[col].astype('category')

    # Convert the DataFrame to a dictionary format for DictVectorizer
    data_dicts = df.to_dict(orient='records')

    # Apply the DictVectorizer to encode categorical variables (One-Hot Encoding)
    transformed_data = dict_vectorizer.transform(data_dicts)

    # Convert the transformed data back to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=dict_vectorizer.get_feature_names_out())

    # Identify the numerical columns that should be scaled (after one-hot encoding)
    # These are the columns in the original data that are numeric (before encoding)
    numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']


    # Apply Min-Max scaling to the numerical columns
    if len(numerical_columns) > 0:
        # Apply the saved Min-Max scaler to the numerical columns
        transformed_df[numerical_columns] = minmax_scaler.transform(transformed_df[numerical_columns])

    # Reindex the transformed data to ensure it has all the features the model expects
    transformed_df = transformed_df.reindex(columns=best_model.feature_names_in_, fill_value=0)

    # Final preprocessed df, ready for modeling
    print("Preprocessed input data, as required by model:")
    print(transformed_df)

    return transformed_df



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

        # Predict using the loaded model
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

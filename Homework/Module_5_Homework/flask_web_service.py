import os
import pickle
from flask import Flask, request, jsonify
import numpy as np


# Initialize Flask app
app = Flask(__name__)

# Change to the directory where the model files are located
os.chdir("/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_5_Homework")

# Load the DictVectorizer
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

# Load the Logistic Regression model
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get client data from the request
    client_data = request.json

    # Transform the client data using the DictVectorizer
    X_new = dv.transform([client_data])

    # Predict the probability of subscription
    probability = model.predict_proba(X_new)[0, 1]  # Get the probability of the positive class

    # Return the probability as JSON
    return jsonify({'probability': probability})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

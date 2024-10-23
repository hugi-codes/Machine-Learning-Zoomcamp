import pandas as pd
import os
import pickle
import numpy as np


os.chdir("/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_5_Homework")
df = pd.read_csv("https://github.com/alexeygrigorev/datasets/raw/refs/heads/master/bank-full.csv"
                 , sep = ";") 

# ===============================================================================================
# Q3: scoring the client
# Load the DictVectorizer
with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

# Load the Logistic Regression model
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

# Define the client data
client_data = {"job": "management", "duration": 400, "poutcome": "success"}

# Transform the client data using the DictVectorizer
X_new = dv.transform([client_data])

# Predict the probability of subscription
probability = model.predict_proba(X_new)[0, 1]  # Get the probability of the positive class

# Output the probability
print(f"The probability that this client will get a subscription is: {probability:.4f}")
# ===============================================================================================

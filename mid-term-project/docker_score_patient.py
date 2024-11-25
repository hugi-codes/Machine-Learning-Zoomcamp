import requests

# Define the URL of the Flask app's endpoint
url = "http://127.0.0.1:9696/predict"  # Use localhost or 127.0.0.1 to access the container

# Input data to send in the POST request
input_data = {
    "Age": 45,
    "Sex": "M",  # M: Male, F: Female
    "ChestPainType": "NAP",  # TA, ATA, NAP, ASY
    "RestingBP": 120,  # Resting blood pressure [mm Hg]
    "Cholesterol": 230,  # Serum cholesterol [mm/dl]
    "FastingBS": 0,  # 1 if FastingBS > 120 mg/dl, 0 otherwise
    "RestingECG": "Normal",  # Normal, ST, LVH
    "MaxHR": 150,  # Numeric value between 60 and 202
    "ExerciseAngina": "N",  # Y: Yes, N: No
    "Oldpeak": 1.5,  # Depression in ST
    "ST_Slope": "Flat"  # Up, Flat, Down
}

# Send the POST request with the input data as JSON
response = requests.post(url, json=input_data)

# Check the response status and print the result
if response.status_code == 200:
    result = response.json()
    print(f"Prediction Probability: {result['prediction_proba']:.4f}")
    print(f"Result: {result['result']}")
else:
    print(f"Error: {response.status_code}, {response.text}")

import requests

url = "http://0.0.0.0:5000/predict"  # Update this URL if hosted on a different machine
client = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=client)

# Check if the request was successful and print the result
if response.status_code == 200:
    result = response.json()
    print(f"The probability that this client will get a subscription is: {result['probability']:.4f}")
else:
    print(f"Error: {response.status_code}, {response.text}")

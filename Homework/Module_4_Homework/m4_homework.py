# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Load data 
file_path = "/home/timhug/Machine-Learning-Zoomcamp/bank-full.csv"
df = pd.read_csv(file_path, sep = ";")

# Selecting relevant columns
columns_to_keep = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

df = df[columns_to_keep]

# ===============================================================================================
# Creating a train, val and test split with scikit learn

# Step 1: Split into train+validation (80%) and test (20%)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

# Step 2: Split the train+validation set into train (60%) and validation (20%)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=1)  # 0.25 * 80% = 20%

# ===============================================================================================
# Q1: AUC scores
# Assuming train_df is the training set and contains both the features and the target 'y'
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
y_train = train_df['y']  # Target variable

# Dictionary to store AUC scores
auc_scores = {}

# Loop through each numerical column
for col in numerical_cols:
    # Compute the initial AUC score
    auc = roc_auc_score(y_train, train_df[col])
    
    # If AUC is less than 0.5, invert the column
    if auc < 0.5:
        auc = roc_auc_score(y_train, -train_df[col])
    
    # Store the AUC score in the dictionary
    auc_scores[col] = auc

# Display the AUC scores for each numerical feature
for col, score in auc_scores.items():
    print(f'{col}: AUC = {score:.4f}')

# ===============================================================================================
# Q2: Training the model
X_train = train_df.drop(columns=['y'])
y_train = train_df['y']

X_val = val_df.drop(columns=['y'])
y_val = val_df['y']

# Step 1: One-hot encoding of categorical variables using DictVectorizer
# Combine categorical and numerical columns into a list of dictionaries
dv = DictVectorizer(sparse=False)

# Convert training and validation data to dictionaries for DictVectorizer
X_train_dict = X_train.to_dict(orient='records')
X_val_dict = X_val.to_dict(orient='records')

# Apply one-hot encoding to the data
X_train_encoded = dv.fit_transform(X_train_dict)
X_val_encoded = dv.transform(X_val_dict)

# Step 2: Train the Logistic Regression model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)

# Step 3: Make predictions on the validation set
y_val_pred = model.predict_proba(X_val_encoded)[:, 1]

# Step 4: Compute the AUC on the validation set
auc_val = roc_auc_score(y_val, y_val_pred)

# Print the AUC score rounded to 3 digits
print(f'Validation AUC: {auc_val:.3f}')

# ===============================================================================================
# Q3: Precision and Recal
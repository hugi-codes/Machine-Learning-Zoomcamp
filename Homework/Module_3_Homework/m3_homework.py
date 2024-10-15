# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the data
df = pd.read_csv("/home/timhug/Machine-Learning-Zoomcamp/Homework/Module_3_Homework/bank-full.csv",
                 delimiter=';')

columns_to_keep = [
    'age', 'job', 'marital', 'education', 'balance', 'housing', 
    'contact', 'day', 'month', 'duration', 'campaign', 
    'pdays', 'previous', 'poutcome', 'y'
]

# Creating a new DataFrame with only the selected columns
df_selected = df[columns_to_keep]

# Checking for missing values
missing_values = df_selected.isnull().sum()

# ===============================================================================
# Q1: Find the mode (most frequent observation) of the 'education' column
education_mode = df['education'].mode()
# Answer: "secondary" is the most common category in the education column.

# ===============================================================================
# Q2: Create correlation matrix for all numeric features
# Step 1: Select only numerical features
df_numerical = df.select_dtypes(include=['int64', 'float64'])

# Step 2: Compute the correlation matrix
correlation_matrix = df_numerical.corr()

# ===============================================================================
# Target encoding
df_selected['y'] = df_selected['y'].replace({'yes': 1, 'no': 0})


# Target encoding
# Step 1: Separate the features and the target variable
X = df_selected.drop('y', axis=1)  # Features
y = df_selected['y']               # Target variable

# Step 2: Split the data into training (60%) and temporary dataset (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 3: Split the temporary dataset into validation (20%) and test (20%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of the resulting datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ===============================================================================
# Q3: Mutual information score
# Step 1: Create df_categorical containing only categorical features (string dtype)
df_categorical = X_train.select_dtypes(include=['object'])

def calculate_mi(series):
    return mutual_info_score(series, y_train)

df_mi = df_categorical.apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
df_mi['MI'] = df_mi['MI'].round(2)

# ===============================================================================
# Q4: Training a logistic regression model
# Separate the features and target variable
X = df_selected.drop('y', axis=1)  # Features
y = df_selected['y']               # Target variable

# Split the dataset into training and validation sets for both features and target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training and validation feature sets into dictionaries
train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

# Use DictVectorizer to one-hot encode categorical features and keep numerical ones as they are
dv = DictVectorizer(sparse=False)

# Fit and transform the training data
X_train = dv.fit_transform(train_dict)

# Transform the validation data
X_val = dv.transform(val_dict)

# Logistic regression model with specified parameters
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the validation dataset
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

# Print accuracy rounded to 2 decimal digits
print(f"Validation Accuracy: {accuracy:.2f}")

# ===============================================================================
# Q5:  Feature elimination
# Separate the features and target variable
X = df_selected.drop('y', axis=1)  # Features
y = df_selected['y']               # Target variable

# Split the dataset into training and validation sets for both features and target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training and validation feature sets into dictionaries
train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

# Use DictVectorizer to one-hot encode categorical features and keep numerical ones as they are
dv = DictVectorizer(sparse=False)

# Fit and transform the training data
X_train_transformed = dv.fit_transform(train_dict)
X_val_transformed = dv.transform(val_dict)

# Logistic regression model with specified parameters
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Fit the model on the full training data
model.fit(X_train_transformed, y_train)

# Predict on the validation dataset
y_pred = model.predict(X_val_transformed)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_val, y_pred)
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# List of features to test for elimination
features_to_test = ['age', 'balance', 'marital', 'previous']

# Store differences in accuracy
feature_differences = {}

# Iterate over the features to eliminate them one by one
for feature in features_to_test:
    
    # Remove the feature from the original training and validation data
    X_train_dropped = X_train.drop(columns=[feature])
    X_val_dropped = X_val.drop(columns=[feature])
    
    # Convert the training and validation feature sets without the feature into dictionaries
    train_dict_dropped = X_train_dropped.to_dict(orient='records')
    val_dict_dropped = X_val_dropped.to_dict(orient='records')
    
    # Transform the data using the same DictVectorizer
    X_train_transformed_dropped = dv.transform(train_dict_dropped)
    X_val_transformed_dropped = dv.transform(val_dict_dropped)
    
    # Train a logistic regression model without the feature
    model.fit(X_train_transformed_dropped, y_train)
    
    # Predict on the validation dataset without the feature
    y_pred_dropped = model.predict(X_val_transformed_dropped)
    
    # Calculate accuracy without the feature
    accuracy_dropped = accuracy_score(y_val, y_pred_dropped)
    
    # Calculate the difference in accuracy
    accuracy_difference = baseline_accuracy - accuracy_dropped
    feature_differences[feature] = accuracy_difference
    
    print(f"Feature: {feature}, Accuracy without feature: {accuracy_dropped:.4f}, Difference: {accuracy_difference:.4f}")

# Find the feature with the smallest difference
least_useful_feature = min(feature_differences, key=feature_differences.get)

print(f"The least useful feature is: {least_useful_feature}")

# ===============================================================================
# Q6:  Train regularized linear regression model
# Separate the features and target variable
X = df_selected.drop('y', axis=1)  # Features
y = df_selected['y']               # Target variable

# Split the dataset into training and validation sets for both features and target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training and validation feature sets into dictionaries
train_dict = X_train.to_dict(orient='records')
val_dict = X_val.to_dict(orient='records')

# Use DictVectorizer to one-hot encode categorical features and keep numerical ones as they are
dv = DictVectorizer(sparse=False)

# Fit and transform the training data
X_train_transformed = dv.fit_transform(train_dict)
X_val_transformed = dv.transform(val_dict)

# Values of C to test
C_values = [0.01, 0.1, 1, 10, 100]

# Store accuracies for different values of C
accuracies = {}

# Iterate over the different values of C
for C in C_values:
    
    # Logistic regression model with the current value of C and specified parameters
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    
    # Fit the model on the training data
    model.fit(X_train_transformed, y_train)
    
    # Predict on the validation dataset
    y_pred = model.predict(X_val_transformed)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    # Round accuracy to 3 decimal digits
    rounded_accuracy = round(accuracy, 3)
    
    # Store accuracy for the current value of C
    accuracies[C] = rounded_accuracy
    
    print(f"C: {C}, Validation Accuracy: {rounded_accuracy}")

# Find the value of C that gives the best accuracy
best_C = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_C]

print(f"Best C: {best_C}, Best Validation Accuracy: {best_accuracy}")
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
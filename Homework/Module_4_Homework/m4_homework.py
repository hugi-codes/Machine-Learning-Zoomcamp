# Importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Loading data 
file_path = "/home/timhug/Machine-Learning-Zoomcamp/bank-full.csv"
df = pd.read_csv(file_path, sep = ";")

# Converting target variable 'y' to binary 0 (no) and 1 (yes)
df['y'] = df['y'].map({'yes': 1, 'no': 0})


# Selecting relevant columns
columns_to_keep = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

df = df[columns_to_keep]

# ===============================================================================================
# Creating a train, val and test split with scikit learn

# Splitting into train+validation (80%) and test (20%)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

# Splitting the train+validation set into train (60%) and validation (20%)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=1)  # 0.25 * 80% = 20%

# ===============================================================================================
# Q1: AUC scores

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
y_train = train_df['y']  # Target variable

# Dictionary to store AUC scores
auc_scores = {}

# Looping through each numerical column
for col in numerical_cols:
    # Computing the initial AUC score
    auc = roc_auc_score(y_train, train_df[col])
    
    # If AUC is less than 0.5, invert the column
    if auc < 0.5:
        auc = roc_auc_score(y_train, -train_df[col])
    
    # Storing the AUC score in the dictionary
    auc_scores[col] = auc

# Displaying the AUC scores for each numerical feature
for col, score in auc_scores.items():
    print(f'{col}: AUC = {score:.4f}')

# ===============================================================================================
# Q2: Training the model
X_train = train_df.drop(columns=['y'])
y_train = train_df['y']

X_val = val_df.drop(columns=['y'])
y_val = val_df['y']

# One-hot encoding of categorical variables using DictVectorizer
# Combining categorical and numerical columns into a list of dictionaries
dv = DictVectorizer(sparse=False)

# Converting training and validation data to dictionaries for DictVectorizer
X_train_dict = X_train.to_dict(orient='records')
X_val_dict = X_val.to_dict(orient='records')

# Applying one-hot encoding to the data
X_train_encoded = dv.fit_transform(X_train_dict)
X_val_encoded = dv.transform(X_val_dict)

# Training the Logistic Regression model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)

# Making predictions on the validation set
y_val_pred = model.predict_proba(X_val_encoded)[:, 1]

# Computing the AUC on the validation set
auc_val = roc_auc_score(y_val, y_val_pred)

print(f'Validation AUC: {auc_val:.3f}')

# ===============================================================================================
# Q3: Precision and Recall
# Making probability predictions on the validation set
y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]

# Defining the thresholds from 0.0 to 1.0 with a step of 0.01
thresholds = np.arange(0.0, 1.01, 0.01)

# Lists to store precision and recall values
precisions = []
recalls = []

# Computing precision and recall for each threshold
for threshold in thresholds:
    # Converting probabilities to binary predictions based on the threshold
    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
    
    # Computing precision and recall
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    # Storing the results
    precisions.append(precision)
    recalls.append(recall)

# Plotting precision and recall curves
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions, label='Precision', color='blue')
plt.plot(thresholds, recalls, label='Recall', color='red')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs. Threshold')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('precision_recall_plot.png', dpi=300)  # Saving the plot as a PNG file with 300 DPI

# ===============================================================================================
# Q4: F1 Score

# List to store F1 scores
f1_scores = []

# Computing F1 score for each threshold
for threshold in thresholds:
    # Converting probabilities to binary predictions based on the threshold
    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
    
    # Computing F1 score
    f1 = f1_score(y_val, y_val_pred)
    f1_scores.append(f1)

# Finding the threshold where F1 is maximal
max_f1_index = np.argmax(f1_scores)  # Finding the index of the maximum F1 score
best_threshold = thresholds[max_f1_index]  # Getting the corresponding threshold

print(f'Threshold where F1 is maximal: {best_threshold:.2f}')
print(f'Maximal F1 score: {f1_scores[max_f1_index]:.3f}')

# ===============================================================================================
# Q5: K-Fold Cross Validation

# Preparing the full training data
X_full = train_val_df.drop(columns=['y'])  # Full dataset without target
y_full = train_val_df['y']                   # Target variable

# Initializing KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# List to store AUC scores for each fold
auc_scores = []

# Iterating over K-Folds
for train_index, val_index in kf.split(X_full):
    # Split the data
    X_train_fold, X_val_fold = X_full.iloc[train_index], X_full.iloc[val_index]
    y_train_fold, y_val_fold = y_full.iloc[train_index], y_full.iloc[val_index]
    
    # One-hot encoding of categorical variables using DictVectorizer
    dv = DictVectorizer(sparse=False)
    
    # Convert training and validation data to dictionaries for DictVectorizer
    X_train_dict = X_train_fold.to_dict(orient='records')
    X_val_dict = X_val_fold.to_dict(orient='records')

    # Apply one-hot encoding to the data
    X_train_encoded = dv.fit_transform(X_train_dict)
    X_val_encoded = dv.transform(X_val_dict)
    
    # Train the model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_encoded, y_train_fold)
    
    # Make predictions and evaluate AUC
    y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
    auc = roc_auc_score(y_val_fold, y_val_pred_proba)
    auc_scores.append(auc)

# Calculate the standard deviation of the AUC scores
std_dev_auc = np.std(auc_scores)

# Print the results
print(f'AUC scores across folds: {auc_scores}')
print(f'Standard deviation of AUC scores: {std_dev_auc:.4f}')

# ===============================================================================================
# Q5: Hyperparameter tuning

# Defining the C values to test
C_values = [0.000001, 0.001, 1]

# Initializing KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Dictionary to store mean AUC scores and their standard deviations for each C value
results = {}

# Iterate over the C values
for C in C_values:
    auc_scores = []  # List to store AUC scores for each fold
    
    # Iterating over K-Folds
    for train_index, val_index in kf.split(X_full):
        # Splitting the data
        X_train_fold, X_val_fold = X_full.iloc[train_index], X_full.iloc[val_index]
        y_train_fold, y_val_fold = y_full.iloc[train_index], y_full.iloc[val_index]
        
        # One-hot encoding of categorical variables using DictVectorizer
        dv = DictVectorizer(sparse=False)
        
        # Converting training and validation data to dictionaries for DictVectorizer
        X_train_dict = X_train_fold.to_dict(orient='records')
        X_val_dict = X_val_fold.to_dict(orient='records')

        # Applying one-hot encoding to the data
        X_train_encoded = dv.fit_transform(X_train_dict)
        X_val_encoded = dv.transform(X_val_dict)
        
        # Training the model
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train_fold)
        
        # Making predictions and evaluate AUC
        y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
        auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        auc_scores.append(auc)
    
    # Computing mean and std of AUC scores for the current C value
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    # Storing results in the dictionary with rounded values
    results[C] = (round(mean_auc, 3), round(std_auc, 3))

# Printing results for each C value
for C, (mean_auc, std_auc) in results.items():
    print(f'C = {C}: Mean AUC = {mean_auc}, Std AUC = {std_auc}')

# Determining the best C value based on mean AUC
best_C = max(results, key=lambda x: results[x][0])  # Find the C with the maximum mean AUC
print(f'The best C value is: {best_C} with Mean AUC = {results[best_C][0]}')
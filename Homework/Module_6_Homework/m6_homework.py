import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Loading the data 
url = 'https://github.com/alexeygrigorev/datasets/raw/refs/heads/master/jamb_exam_results.csv'
df = pd.read_csv(url)

# The goal of this homework is to create a regression model 
# for predicting the performance of students on a standardized test (column 'JAMB_Score').

# Col names to lower case
df.columns = df.columns.str.lower().str.replace(' ', '_')

# ========================================================================================0
# Data prep
# Remove the 'student_id' column
df.drop(columns=['student_id'], inplace=True)

# Fill missing values with zeros
df.fillna(0, inplace=True)

# Split the data into train, validation, and test sets (60%/20%/20%)
# Use train_test_split twice to achieve the required splits
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=1)  # 60% train, 40% temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=1)  # 20% validation, 20% test

# Convert the dataframes into matrices using DictVectorizer
vectorizer = DictVectorizer(sparse=True)

# Convert the dataframes to dictionaries for DictVectorizer
train_matrix = vectorizer.fit_transform(train_df.to_dict(orient='records'))
val_matrix = vectorizer.transform(val_df.to_dict(orient='records'))
test_matrix = vectorizer.transform(test_df.to_dict(orient='records'))

# Print shapes of the resulting matrices
print("Train matrix shape:", train_matrix.shape)
print("Validation matrix shape:", val_matrix.shape)
print("Test matrix shape:", test_matrix.shape)
# ========================================================================================
# Q1:  Feature used for splitting if depth = 1

# Extract the target variable 'jamb_score'
X_train = train_df.drop(columns=['jamb_score'])
y_train = train_df['jamb_score']

# Convert the features into matrices using the previously fitted DictVectorizer
X_train_matrix = vectorizer.transform(X_train.to_dict(orient='records'))

# Train a Decision Tree Regressor with max_depth=1
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train_matrix, y_train)

# Display the feature used for splitting
feature_names = vectorizer.get_feature_names_out()
important_feature_index = np.argmax(dt.feature_importances_)
print("Feature used for splitting:", feature_names[important_feature_index])

# ========================================================================================
# Q2: Training an RF regressor and obtaining RMSE on validation data

# Extract features and target for training and validation
X_train = train_df.drop(columns=['jamb_score'])
y_train = train_df['jamb_score']
X_val = val_df.drop(columns=['jamb_score'])
y_val = val_df['jamb_score']

# Convert the features into matrices using the previously fitted DictVectorizer
X_train_matrix = vectorizer.transform(X_train.to_dict(orient='records'))
X_val_matrix = vectorizer.transform(X_val.to_dict(orient='records'))

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train_matrix, y_train)

# Predict on the validation data
y_val_pred = rf.predict(X_val_matrix)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("RMSE on the validation data:", rmse)

# ========================================================================================
# Q3: Experimenting with n_estimator parameter and obtaining respective RMSEs

# Prepare to store the RMSE values for different n_estimators
rmse_values = []

# Loop through different n_estimators from 10 to 200 with a step of 10
for n in range(10, 201, 10):
    # Train a Random Forest Regressor with the current n_estimators
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train_matrix, y_train)
    
    # Predict on the validation data
    y_val_pred = rf.predict(X_val_matrix)
    
    # Calculate the RMSE and store it
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_values.append((n, rmse))

# Print the results
for n, rmse in rmse_values:
    print(f"n_estimators={n}, RMSE={rmse:.3f}")

# Find the value after which RMSE stops improving
min_rmse = min(rmse_values, key=lambda x: x[1])
print(f"\nLowest RMSE of {min_rmse[1]:.3f} occurs at n_estimators={min_rmse[0]}")

# ========================================================================================
# Q4: Trying different values of max_depth parameter

# Prepare to store the mean RMSE for each max_depth and n_estimators combination
results = []

# List of max_depth values to test
max_depth_values = [10, 15, 20, 25]

# Loop over each max_depth value
for max_depth in max_depth_values:
    rmse_list = []
    
    # Loop through different n_estimators from 10 to 200 with step 10
    for n in range(10, 201, 10):
        # Initialize the RandomForestRegressor with the current max_depth and n_estimators
        rf = RandomForestRegressor(n_estimators=n, max_depth=max_depth, random_state=1, n_jobs=-1)
        
        # Train the model
        rf.fit(X_train_matrix, y_train)
        
        # Predict on the validation data
        y_val_pred = rf.predict(X_val_matrix)
        
        # Calculate RMSE for this combination of max_depth and n_estimators
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_list.append(rmse)
    
    # Calculate the mean RMSE for the current max_depth
    mean_rmse = np.mean(rmse_list)
    results.append((max_depth, mean_rmse))

# Find the best max_depth based on the lowest mean RMSE
best_max_depth = min(results, key=lambda x: x[1])
print(f"The best max_depth is {best_max_depth[0]} with a mean RMSE of {best_max_depth[1]:.3f}")


# ========================================================================================
# Q5: Extracting feature importance

# Train the RandomForestRegressor with the specified parameters
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train_matrix, y_train)

# Get feature importance information
feature_importances = rf.feature_importances_

# Get the feature names from the DictVectorizer
feature_names = vectorizer.get_feature_names_out()

# Create a list of feature names and their corresponding importance scores
importance_data = list(zip(feature_names, feature_importances))

# Sort the features by importance (from highest to lowest)
importance_data_sorted = sorted(importance_data, key=lambda x: x[1], reverse=True)

# Print the most important feature and its importance
most_important_feature = importance_data_sorted[0]
print(f"The most important feature is '{most_important_feature[0]}' with an importance score of {most_important_feature[1]:.4f}")

# ========================================================================================
# Q6: ETraining XGBoost

# Prepare the data by converting into DMatrix format
dtrain = xgb.DMatrix(X_train_matrix, label=y_train)
dval = xgb.DMatrix(X_val_matrix, label=y_val)

# Create the watchlist to track the training and validation metrics
watchlist = [(dtrain, 'train'), (dval, 'eval')]

# Define the xgb_params
xgb_params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Train the model with eta = 0.3
xgb_params['eta'] = 0.3
bst_03 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)

# Calculate RMSE on the validation set for eta=0.3
y_val_pred_03 = bst_03.predict(dval)
rmse_03 = np.sqrt(mean_squared_error(y_val, y_val_pred_03))
print(f"RMSE with eta=0.3: {rmse_03:.4f}")

# Train the model with eta = 0.1
xgb_params['eta'] = 0.1
bst_01 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist)

# Calculate RMSE on the validation set for eta=0.1
y_val_pred_01 = bst_01.predict(dval)
rmse_01 = np.sqrt(mean_squared_error(y_val, y_val_pred_01))
print(f"RMSE with eta=0.1: {rmse_01:.4f}")

# Compare RMSE scores and determine the best eta
if rmse_03 < rmse_01:
    print("eta=0.3 gives the best RMSE.")
else:
    print("eta=0.1 gives the best RMSE.")
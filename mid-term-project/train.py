# The following code is notebook.ipynb exported as a Python script
# Note: code for EDA has been omitted. Only code necessary for model training has been kept from notebook.ipynb
# Exporting notebook.ipynb to train.py is a requirement for mid-term project. 
# For more info on the project requirements, please check the links provided in README.md

# %% [markdown]
# ## Importing packages

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn
import kagglehub


# %% [markdown]
# ## Loading the data

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

print("Path to dataset files:", path)

# %%
import pandas as pd

# Specify the path to the CSV file
file_path = '/home/timhug/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)


# %% [markdown]
# ## Data type conversions

# %%
# Transform relevant columns to categorical
categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']
df[categorical_columns] = df[categorical_columns].astype('category')


# %%
# One-hot encoding of categorical features

# Identify categorical columns (e.g., dtype == object or category)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Perform one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# %%
# Min Max scaling of numerical features
# This is not necessary for random forest, but for other algorithms which are distance-based
# Some algos have different requirements for preprocessing than others 
from sklearn.preprocessing import MinMaxScaler

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Apply Min-Max Scaling
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Scales All Features to the Same Range: Each feature's values are normalized to lie within the specified range, commonly [0, 1].


# %%
# Creating Feature Matrix X and target var series y
X = df.iloc[ : , :-1]
y = df.iloc[ : , -1]

# %%


# %% [markdown]
# ## Modeling: Mutltiple models and hyper parameter tuning

# %%
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd

# Set the random seed for reproducibility
random_seed = 42

# Split the dataset into training and hold-out test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Initialize the models and their parameter grids with larger values
models = {
    "SVC": SVC(random_state=random_seed),
    "Random Forest": RandomForestClassifier(random_state=random_seed),
    "Logistic Regression": LogisticRegression(random_state=random_seed),
    "XGBoost": xgb.XGBClassifier(seed=random_seed)  # In XGBoost, the random seed is set via `seed`
}

param_grids = {
    "SVC": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    },
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6]
    }
}

# Initialize a list to store all results
all_results = []

# Define the cross-validation strategy with random_state
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=random_seed)

# Iterate over each model and its corresponding grid of hyperparameters
for model_name, model in models.items():
    print(f"Training {model_name}...\n")
    
    # Get the parameter grid for the current model
    param_grid = param_grids[model_name]
    
    # Perform GridSearchCV with 5-fold cross-validation 
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)  # Fit on the training data
    
    # Calculate the number of fits
    k_folds = 5
    n_param_combinations = 1
    for value in param_grid.values():
        n_param_combinations *= len(value)
    n_fits = k_folds * n_param_combinations
    
    # Print the number of fits for the current model
    print(f"Number of fits for {model_name}: {n_fits}")
    
    # Extract the results for the current grid search
    cv_results = grid_search.cv_results_
    
    # Store the results along with test set accuracy
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Iterate through the results and store the test accuracy and corresponding hyperparameters
    for mean_test_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        all_results.append({
            'model': model_name,
            'mean_cv_accuracy': mean_test_score,
            'test_accuracy': test_accuracy,  # Store the test accuracy
            'hyperparameters': params
        })
    
    # Print the best score and best hyperparameters for the current model
    print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_:.4f}")
    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    print(f"Test set accuracy for {model_name}: {test_accuracy:.4f}")
    print("\n" + "="*50)

# Sort all results by test accuracy in descending order
sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)

# Get the overall winner (model with the highest test accuracy)
best_result = sorted_results[0]
print("\nOverall Winner (based on test set accuracy):")
print(f"Model: {best_result['model']}")
print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
print(f"Best Hyperparameters: {best_result['hyperparameters']}")


# %%
# Now training the best model on the entire data 

# Identify the best model based on the sorted results
best_model_name = best_result['model']
best_hyperparameters = best_result['hyperparameters']

# Reinitialize the best model with the optimal hyperparameters
if best_model_name == "SVC":
    best_model = SVC(**best_hyperparameters, random_state=random_seed)
elif best_model_name == "Random Forest":
    best_model = RandomForestClassifier(**best_hyperparameters, random_state=random_seed)
elif best_model_name == "Logistic Regression":
    best_model = LogisticRegression(**best_hyperparameters, random_state=random_seed)
elif best_model_name == "XGBoost":
    best_model = xgb.XGBClassifier(**best_hyperparameters, seed=random_seed)
else:
    raise ValueError(f"Unknown model name: {best_model_name}")

# Train the best model on the entire dataset (X, y)
print(f"\nTraining the best model ({best_model_name}) on the full dataset...")
best_model.fit(X, y)



# %%
# Saving the mbest model that has been trained on the entire data
# Save the trained model as a pickle file
import pickle
import os

# Define the file name for the pickle file
pickle_filename = f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"

# Get the directory of the current Python file (where the script is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to save the pickle file in the same directory as the script
pickle_filepath = os.path.join(script_dir, pickle_filename)

# Save the trained model as a pickle file
with open(pickle_filepath, 'wb') as file:
    pickle.dump(best_model, file)

print(f"\nThe best model ({best_model_name}) has been saved as '{pickle_filepath}'.")
print("Finished running train.py")

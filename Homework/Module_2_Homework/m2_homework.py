# %%
# Import packages
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import seaborn as sns
import matplotlib.pyplot as plt


# Reading the data
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv"
df = pd.read_csv(url)

# Normalizing column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Selecting the relevant columns (according to the homework)
new_df = df[['ram', 'storage', 'screen', 'final_price']]

# Histogram of final_price
plt.figure(figsize=(8, 6))
sns.histplot(new_df['final_price'], kde=True)  # kde=True adds the KDE curve along with the histogram
plt.title('Distribution of Final Price')
plt.xlabel('Final Price')
plt.ylabel('Frequency')
plt.savefig('final_price_distribution.png')
plt.show()

# Q1: Identifying the column with missing values
new_df.isnull().sum()

# Q1: Median of column "ram"
new_df['ram'].median()

# ===============================================================================
# Preparing the dataset: shuffling and splitting

# Shuffling
n = len(new_df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

# Splitting  data in train/val/test sets, with 60%/20%/20% distribution
# Printing any of the vars below, we can see that the index of each respective df is shuffled.
df_train = new_df.iloc[idx[:n_train]]
df_val = new_df.iloc[idx[n_train:n_train+n_val]]
df_test = new_df.iloc[idx[n_train+n_val:]]

# Dropping indeces
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Q3: Training linear regression without regularization, using two options:
# Option 1: Fill missing values with 0
# Option 2: Fill missing values with the mean of the training set

# Extracting features and target variable from the training set
X_train = df_train[['screen', 'ram', 'storage']].values  # Feature variable (as a numpy array)
y_train = df_train['final_price'].values  # Target variable (as a numpy array)

# Extracting features and target variable from the validation set
X_val = df_val[['screen', 'ram', 'storage']].values  # Feature variable for validation
y_val = df_val['final_price'].values  # Target variable for validation

# Define the linear regression training function
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])  # Create a column of ones for the intercept
    X = np.column_stack([ones, X])  # Add the column of ones to X

    XTX = X.T.dot(X)  # Compute X^T * X
    XTX_inv = np.linalg.inv(XTX)  # Compute the inverse of X^T * X
    w_full = XTX_inv.dot(X.T).dot(y)  # Compute the weights
    
    return w_full[0], w_full[1:]  # Return the intercept and weights

# Define the RMSE function
def rmse(y, y_pred):
    se = (y - y_pred) ** 2  # Calculate squared errors
    mse = se.mean()          # Calculate mean squared error
    return np.sqrt(mse)     # Return root mean squared error

# Option 1: Fill missing values with 0
X_train_0 = np.nan_to_num(X_train)  # Fill NaNs with 0
X_val_0 = np.nan_to_num(X_val)  # Fill NaNs with 0

# Train the linear regression model
w0_0, w_0 = train_linear_regression(X_train_0, y_train)

# Predictions for validation set with model 0
y_pred_0 = w0_0 + X_val_0.dot(w_0)

# Calculate RMSE for Option 1 using the custom rmse function
rmse_0 = rmse(y_val, y_pred_0)

# Option 2: Fill missing values with the mean of the training set
mean_screen = np.nanmean(X_train)  # Calculate the mean ignoring NaNs
X_train_mean = np.where(np.isnan(X_train), mean_screen, X_train)  # Fill NaNs with mean
X_val_mean = np.where(np.isnan(X_val), mean_screen, X_val)  # Fill NaNs with mean

# Train the linear regression model
w0_mean, w_mean = train_linear_regression(X_train_mean, y_train)

# Predictions for validation set with model mean
y_pred_mean = w0_mean + X_val_mean.dot(w_mean)

# Calculate RMSE for Option 2 using the custom rmse function
rmse_mean = rmse(y_val, y_pred_mean)

# Print RMSE results rounded to 2 decimal digits
print(f"RMSE with missing values filled with 0: {round(rmse_0, 2)}")
print(f"RMSE with missing values filled with mean: {round(rmse_mean, 2)}")
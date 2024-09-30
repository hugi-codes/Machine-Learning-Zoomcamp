# Import packages
import pandas as pd
import numpy as np

# Reading the data
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv"
df = pd.read_csv(url)

# Q1: Pandas vesion
pd.__version__

# Q2: number of rows
print(df.shape[0])

# Q3: number of laptop brands
df["Brand"].nunique()

# Q4: number of columns with missing values
missing_columns = df.isnull().sum()  # Counts missing values per column
num_missing_columns = (missing_columns > 0).sum()  # Counts columns with at least one missing value

print(num_missing_columns)

# Q5: maximum final price
# Filter for Dell notebooks
dell_notebooks = df[df['Brand'] == 'Dell']

# Find the maximum final price
max_final_price = dell_notebooks['Final Price'].max()

print(max_final_price)


# Q6: median value of screen
# Find the median value of the 'Screen' column
median_screen_before = df['Screen'].median()
print("Median value of 'Screen' before filling missing values:", median_screen_before)

# Find the most frequent value (mode) of the 'Screen' column
most_frequent_screen = df['Screen'].mode()[0]
print("Most frequent value (mode) of 'Screen':", most_frequent_screen)

# Fill missing values in 'Screen' column with the most frequent value
df['Screen'].fillna(most_frequent_screen, inplace=True)

# Find the median value of the 'Screen' column again after filling missing values
median_screen_after = df['Screen'].median()
print("Median value of 'Screen' after filling missing values:", median_screen_after)


# Q7: Sum of weights
# Step 1: Select all "Innjoo" laptops and filter columns
innjoo_laptops = df[df['Brand'] == 'Innjoo'][['RAM', 'Storage', 'Screen']]

# Step 2: Get the underlying NumPy array X
X = innjoo_laptops.to_numpy()

# Step 3: Compute matrix-matrix multiplication X^T * X
XTX = np.dot(X.T, X)

# Step 4: Compute the inverse of XTX
XTX_inv = np.linalg.inv(XTX)

# Step 5: Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100])

# Multiply the inverse of XTX with X^T and then with y
w = np.dot(XTX_inv, np.dot(X.T, y))

# Step 6: Calculate the sum of all elements in w
result_sum = np.sum(w)
print("Sum of all elements in the result w:", result_sum)

import pandas as pd

# Loading the data 
url = 'https://github.com/alexeygrigorev/datasets/raw/refs/heads/master/jamb_exam_results.csv'
df = pd.read_csv(url)


# The goal of this homework is to create a regression model 
# for predicting the performance of students on a standardized test (column 'JAMB_Score').

# Col names to lower case
df.columns = df.columns.str.lower().str.replace(' ', '_')
df
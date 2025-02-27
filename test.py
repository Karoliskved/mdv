import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("WineQT.csv")

# Select a numerical column (replace 'column_name' with your actual column)
column = df['residual sugar']

# Variance
variance = column.var()

# Standard Deviation
std_dev = column.std()

# Range
data_range = column.max() - column.min()

# Interquartile Range (IQR)
iqr = column.quantile(0.75) - column.quantile(0.25)




print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
print(f"Range: {data_range}")
print(f"Interquartile Range (IQR): {iqr}")


print(df.quality.value_counts())



# Reshape data for sklearn (expects 2D array for X)
X = df[['alcohol']]  # Double brackets keep it 2D
Y = df['quality']

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Get regression coefficients
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"R-squared: {model.score(X, Y)}")  # Model performance
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import radviz
import seaborn as sns

import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from plotly.subplots import make_subplots

import plotly.express as px


df = pd.read_csv("WineQT.csv")

df = df.drop("Id", axis=1)





dispersion_metrics = df.describe().T  # Transpose for better readability

# Add custom calculations
dispersion_metrics["range"] = dispersion_metrics["max"] - dispersion_metrics["min"]
dispersion_metrics["IQR"] = dispersion_metrics["75%"] - dispersion_metrics["25%"]

# Display the results
print(dispersion_metrics[[ "min", "max", "range", ]])


dispersion_metrics.to_csv("dispersion_metrics.csv")


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



X = df.drop(columns=['quality'])  

feature_names = X.columns


X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=X.shape[1])  # full PCA
X_pca = pca.fit_transform(X_std)



explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Variance explained by each component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.2%}")

print(f"\nTotal variance in first two components: {cumulative_variance[1]:.2%}")



plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['quality'], cmap='plasma', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - First Two Principal Components')
plt.colorbar(label='Wine Quality')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_wine_scatter.png')
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 2], c=df['quality'], cmap='plasma', edgecolor='k', s=50)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[2])
plt.title(f'Direct Feature Plot: {feature_names[0]} vs {feature_names[2]}')
plt.colorbar(label='Wine Quality')
plt.grid(True)
plt.tight_layout()
plt.savefig('direct_wine_scatter.png')
plt.show()
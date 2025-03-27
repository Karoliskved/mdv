import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import radviz
import seaborn as sns

import plotly.graph_objects as go


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



#df['quality'] = df['quality'].apply(lambda x: 1 if x >= threshold else 0)
df_filtered=df[df['quality'].isin([6, 4])]
plt.figure(figsize=(6,6))
radviz(df_filtered, class_column='quality', alpha=0.5, s=100, color=sns.color_palette("husl", len(df_filtered['quality'].unique())))
plt.show()

scatter_matrix(df, 0.2,  figsize = (10, 10))
plt.show()

scaler = MinMaxScaler()
badWine = df.loc[df["quality"]==3].index
mediocreWine =df.loc[df["quality"]==6].index
goodWine =df.loc[df["quality"]==8].index
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.drop("quality", axis=1, inplace=True)


"""
matrix = make_subplots(
    rows=3, cols=3,  # 2x2 grid of subplots
    specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
           [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}], 
           [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]],  # Each subplot will be a radar chart
    subplot_titles=['Radar Chart 1', 'Radar Chart 2', 'Radar Chart 3', 'Radar Chart 4']
)

fig = px.line_polar(df, df.iloc[badWine[0]].values, df.columns,  line_close=True )
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()

matrix.add_trace(
        fig.data[0], row=1, col=1
    )


fig = px.line_polar(df, df.iloc[badWine[1]].values, df.columns,  line_close=True )
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()

matrix.add_trace(
        fig.data[0], row=1, col=2
    )


fig = px.line_polar(df, df.iloc[badWine[2]].values, df.columns,  line_close=True )
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()

matrix.add_trace(
        fig.data[0], row=1, col=3
    )


fig = px.line_polar(df, df.iloc[mediocreWine[0]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=2, col=1
    )

fig = px.line_polar(df, df.iloc[mediocreWine[1]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=2, col=2
    )

fig = px.line_polar(df, df.iloc[mediocreWine[2]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=2, col=3
    )

print(df.iloc[goodWine])
fig = px.line_polar(df, df.iloc[goodWine[0]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=3, col=1
    )

fig = px.line_polar(df, df.iloc[goodWine[1]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=3, col=2
    )
fig = px.line_polar(df, df.iloc[goodWine[2]].values, df.columns,  line_close=True)
fig.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Set fixed range
    )
)
#fig.show()
matrix.add_trace(
        fig.data[0], row=3, col=3
    )

matrix.update_layout(
    showlegend=False,  # Disable legend
    title_text='Matrix of Radar Charts',
    polar=dict(
        radialaxis=dict(range=[0, 1])  # Fix range from 0 to 1 for consistency
    ),
    margin=dict(t=50, b=50, l=50, r=50)  # Adjust margin for better spacing
)

for i in range(1, 10):  # Loop through subplots
    matrix.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1])  # Set fixed range for radial axis
        ),
        row=(i - 1) // 2 + 1,  # Calculate row based on index
        col=(i - 1) % 2 + 1  
    )

matrix.show()
"""

# Create a subplot grid (3x3 matrix of radar charts)
matrix = make_subplots(
    rows=3, cols=3, shared_xaxes = 'all',  # 3x3 grid of subplots
    specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
           [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}], 
           [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]],  # Each subplot will be a radar chart
    subplot_titles=['Radar Chart 1', 'Radar Chart 2', 'Radar Chart 3', 
                    'Radar Chart 4', 'Radar Chart 5', 'Radar Chart 6', 
                    'Radar Chart 7', 'Radar Chart 8', 'Radar Chart 9']
)

# Function to create radar plot and add it to the matrix
def create_radar_trace(data, row, col):
    print(data)
    fig = px.line_polar(df, df.iloc[data].values, df.columns, line_close=True)
    fig.update_traces(
        selector=dict(type='scatterpolar'),
        line=dict(color='blue'),
        showlegend=False,
    )

    # Set the radial axis range to be fixed from 0 to 1 for all radar charts
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1])  # Fixed range for consistency
        )
    )

    matrix.add_trace(
        fig.data[0], row=row, col=col
    )

# Add traces to the matrix using the indices
create_radar_trace(badWine[0], 1, 1)
create_radar_trace(badWine[1], 1, 2)
create_radar_trace(badWine[2], 1, 3)

create_radar_trace(mediocreWine[0], 2, 1)
create_radar_trace(mediocreWine[1], 2, 2)
create_radar_trace(mediocreWine[2], 2, 3)

create_radar_trace(goodWine[0], 3, 1)
create_radar_trace(goodWine[1], 3, 2)
create_radar_trace(goodWine[2], 3, 3)

# Update layout for the overall figure
matrix.update_layout(
    showlegend=False,  # Disable legend
    title_text='Matrix of Radar Charts',
    margin=dict(t=50, b=50, l=50, r=50)  # Adjust margin for better spacing
)

# Show the final plot
matrix.show()



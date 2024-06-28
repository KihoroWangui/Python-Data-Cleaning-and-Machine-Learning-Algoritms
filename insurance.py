# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("insurance.csv")

# Data Exploration

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Get dataset information
print("\nDataset information:")
print(df.info())

# Statistics of numerical variables
print("\nStatistics of numerical variables:")
print(df.describe())

# Summary statistics of categorical variables
print("\nSummary statistics of categorical variables:")
print(df.describe(include=['object']))

# Visualization of distribution of medical costs
plt.figure(figsize=(10, 6))
sns.histplot(df['medicalCost'], bins=30, kde=True)
plt.title('Distribution of Medical Costs')
plt.xlabel('Medical Cost')
plt.ylabel('Frequency')
plt.show()

# Pairplot to visualize relationships between numerical variables
sns.pairplot(df)
plt.show()

# Boxplot of medical costs by categorical variables
plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='medicalCost', data=df)
plt.title('Boxplot of Medical Costs by Smoker')
plt.xlabel('Smoker')
plt.ylabel('Medical Cost')
plt.show()

# Correlation Analysis

# Compute the correlation matrix
corr_matrix = df.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


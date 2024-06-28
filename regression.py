import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("insurance.csv")

# Exclude non-numeric columns from the DataFrame
numeric_columns = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_columns.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Simple Linear Regression Models

# Select predictors
predictors = ["age", "bmi", "smoker"]

# Iterate through each predictor and build simple linear regression models
for predictor in predictors:
    X = df[[predictor]]
    y = df["medicalCost"]
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Building the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Making predictions
    y_pred = model.predict(X_test)
    
    # Evaluating model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Displaying results
    print(f"\nPredictor: {predictor}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

# Multivariate Regression Models

# Model with all predictors
X_all = df.drop(columns=["medicalCost"])
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.2, random_state=42)
model_all = LinearRegression()
model_all.fit(X_train_all, y_train_all)
y_pred_all = model_all.predict(X_test_all)

# Model with selected predictors
X_selected = df[predictors]
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model_selected = LinearRegression()
model_selected.fit(X_train_selected, y_train_selected)
y_pred_selected = model_selected.predict(X_test_selected)

# Evaluate multivariate models
mae_all = mean_absolute_error(y_test_all, y_pred_all)
mse_all = mean_squared_error(y_test_all, y_pred_all)
r2_all = r2_score(y_test_all, y_pred_all)

mae_selected = mean_absolute_error(y_test_selected, y_pred_selected)
mse_selected = mean_squared_error(y_test_selected, y_pred_selected)
r2_selected = r2_score(y_test_selected, y_pred_selected)

# Displaying results for multivariate models
print("\nMultivariate Regression Model with All Predictors:")
print(f"Mean Absolute Error: {mae_all}")
print(f"Mean Squared Error: {mse_all}")
print(f"R-squared: {r2_all}")

print("\nMultivariate Regression Model with Selected Predictors:")
print(f"Mean Absolute Error: {mae_selected}")
print(f"Mean Squared Error: {mse_selected}")
print(f"R-squared: {r2_selected}")

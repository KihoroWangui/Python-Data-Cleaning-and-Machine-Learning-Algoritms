import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "insurance.csv" 
data = pd.read_csv(file_path)

# Select top 3 predictors based on correlation
top_predictors = ['bmi', 'age', 'children'] 

# Function to build and evaluate a simple linear regression model
def simple_linear_regression(predictor):
    X = data[[predictor]]
    y = data['medicalCost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Simple Linear Regression with {predictor}:')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print()

# Build and evaluate simple linear regression models
for predictor in top_predictors:
    simple_linear_regression(predictor)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data
file_path = "CensusDB.csv"
data = pd.read_csv(file_path)

# Print the first few rows of the dataframe to verify the structure
print(data.head())

# Clean the column names to remove any leading/trailing spaces
data.columns = data.columns.str.strip()

# Print column names to verify the target column name
print("Column names in the dataset:", data.columns)

# Check if 'income' column exists
if 'income' in data.columns:
    target_column = 'income'
else:
    raise KeyError("The target column 'income' is not found in the dataset.")

# Encode categorical variables and prepare the feature matrix and labels
data = pd.get_dummies(data, drop_first=True)
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}

# Define the parameter grid for SVM
param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

# Initialize the models
lr = LogisticRegression(max_iter=1000)
svm = SVC()

# Perform Grid Search with cross-validation for Logistic Regression
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_

# Perform Grid Search with cross-validation for SVM
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_svm = grid_search_svm.best_estimator_

# Make predictions with the best models
y_pred_lr = best_lr.predict(X_test)
y_pred_svm = best_svm.predict(X_test)

# Evaluate the models
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Optimized Logistic Regression:")
print(f"Best Parameters: {grid_search_lr.best_params_}")
print(f"Accuracy: {accuracy_lr}")
print(classification_report(y_test, y_pred_lr))

print("Optimized SVM:")
print(f"Best Parameters: {grid_search_svm.best_params_}")
print(f"Accuracy: {accuracy_svm}")
print(classification_report(y_test, y_pred_svm))

# Compare with non-optimized models
lr_non_optimized = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_non_optimized = SVC().fit(X_train, y_train)

y_pred_lr_non_optimized = lr_non_optimized.predict(X_test)
y_pred_svm_non_optimized = svm_non_optimized.predict(X_test)

accuracy_lr_non_optimized = accuracy_score(y_test, y_pred_lr_non_optimized)
accuracy_svm_non_optimized = accuracy_score(y_test, y_pred_svm_non_optimized)

print("Non-Optimized Logistic Regression:")
print(f"Accuracy: {accuracy_lr_non_optimized}")
print(classification_report(y_test, y_pred_lr_non_optimized))

print("Non-Optimized SVM:")
print(f"Accuracy: {accuracy_svm_non_optimized}")
print(classification_report(y_test, y_pred_svm_non_optimized))

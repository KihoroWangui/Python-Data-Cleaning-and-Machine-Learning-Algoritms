import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
file_path = "censusDB.csv"  
data = pd.read_csv(file_path, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                                     'hours-per-week', 'native-country', 'label'], na_values=' ?')

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Verify if the data is loaded correctly
print(data.head())

# Prepare features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Encode the target variable
y = y.apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()

# Fit and transform the scaler on training data, transform test data
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Build and train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train[numerical_features], y_train)

# Predict the target variable for the test set
y_pred = svm_model.predict(X_test[numerical_features])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

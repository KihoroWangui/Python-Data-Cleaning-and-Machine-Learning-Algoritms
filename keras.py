import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess the data
file_path = "CensusDB.csv"
data = pd.read_csv(file_path)

# Handling categorical data
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'sex', 'native-country']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Splitting the data into features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training and evaluating MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_classifier.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_scaled)

# Evaluating the MLP classifier
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Classifier Accuracy:", accuracy_mlp)
print("Classification Report for MLP Classifier:")
print(classification_report(y_test, y_pred_mlp))

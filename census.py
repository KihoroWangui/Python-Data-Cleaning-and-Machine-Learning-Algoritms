# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = "CensusDB.csv"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'label']
data = pd.read_csv(file_path, names=columns, na_values=' ?')

# Display the first few rows
print(data.head())

# Check the shape of dataset
print("Shape of the dataset:", data.shape)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Check the data types of each column
print("Data types:\n", data.dtypes)

# Summary statistics
print("Summary statistics:\n", data.describe())

# Import necessary libraries
import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'label']
data = pd.read_csv(url, names=columns, na_values=' ?')

# Display the first few rows of the dataset
print(data.head())

# Check the shape of the dataset
print("Shape of the dataset:", data.shape)

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Check the data types of each column
print("Data types:\n", data.dtypes)

# Summary statistics
print("Summary statistics:\n", data.describe())

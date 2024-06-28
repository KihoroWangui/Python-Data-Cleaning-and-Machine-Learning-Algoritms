# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = "CensusDB.csv"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'label']
data = pd.read_csv(file_path, names=columns, na_values=' ?')

# one-hot encoding
data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 
                                     'occupation', 'relationship', 'sex', 'native-country'])

# Display the first few rows of transformed dataset
print(data.head())

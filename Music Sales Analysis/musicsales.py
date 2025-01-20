import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load file
df = pd.read_csv('music_sales.csv')

# Basic dataset info
print(df.head())
print(df.info())
print(df.describe())

# Missing values and rows
print(df.isnull().sum())

df = df.dropna() 
df['Sales'] = df['Sales'].fillna(0)

# duplicates
df = df.drop_duplicates()
#dataset exploration
# Total sales
total_sales = df['Sales'].sum()
print(f"Total Sales: ${total_sales}")

# Sales by Gigplay
sales_by_Gigplay = df.groupby('Gigplay')['Sales'].sum()
print(sales_by_Gigplay)

# Sales by Gigplay
sales_by_Gigplay.plot(kind='bar', title='Sales by Gigplay')
plt.xlabel('Gigplay')
plt.ylabel('Total Sales')
plt.show()

# Export summary statistics
summary = df.groupby('Gigplay')['Sales'].sum().reset_index()
summary.to_csv('music_sales_analysis.csv', index=False)



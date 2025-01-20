import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load file
df = pd.read_csv('music_sales.xlsx')

# Basic dataset info
print(df.head())
print(df.info())
print(df.describe())

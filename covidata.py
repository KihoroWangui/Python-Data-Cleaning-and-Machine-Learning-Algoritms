import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Function to make API request
def make_api_request():
    url = "https://coronavirus.m.pipedream.net/"  
    response = requests.get(url)
    data = response.json()
    return data

# Make two API requests at different time intervals
data_request1 = make_api_request()
time_interval = timedelta(days=1)  
time.sleep(time_interval.seconds)
data_request2 = make_api_request()

# Print API responses
print("API Response 1:")
print(data_request1)

print("\nAPI Response 2:")
print(data_request2)

# Create DataFrames
df1 = pd.DataFrame(data_request1)  
df2 = pd.DataFrame(data_request2)   
# Add timestamp column
timestamp1 = datetime.now() - time_interval
timestamp2 = datetime.now()
df1['timestamp'] = timestamp1
df2['timestamp'] = timestamp2

# Calculate differences
df2['cases_diff'] = df2['cases'] - df1['cases']

# Combine DataFrames
result_df = pd.concat([df1, df2], ignore_index=True)

# Save as CSV
result_df.to_csv('api_data.csv', index=False)

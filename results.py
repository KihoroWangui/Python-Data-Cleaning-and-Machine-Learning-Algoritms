import pandas as pd
import sqlite3
import time

# Load the dataset
df = pd.read_csv('cleaned_amazon.csv', encoding='latin1')

# Display the first few rows and the column names to confirm the correct names
print(df.head())
print(df.columns)

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load the DataFrame into the SQLite database
df.to_sql('your_table_name', conn, index=False, if_exists='replace')

# Define your SQL query with the correct column names and additional analysis
query = """
SELECT MAX(ISBN) as ISBN, MAX(Image) as Image, Title, Author, MAX(Price) as Price, MAX(Quantity) as Quantity, MAX(Category) as Category
FROM your_table_name
GROUP BY Title, Author
ORDER BY ISBN DESC
LIMIT 50;
"""

# Measure the execution time
start_time = time.time()
result_df = pd.read_sql_query(query, conn)
end_time = time.time()

execution_time = end_time - start_time

# Gather query statistics
query_stats = {
    'query': query,
    'execution_time_seconds': execution_time,
    'number_of_records_returned': len(result_df)
}

# Save the result DataFrame to CSV
result_df.to_csv('query_resultsb.csv', index=False)

# Save the query stats to a CSV file
stats_df = pd.DataFrame([query_stats])
stats_df.to_csv('query_statsb.csv', index=False)

# Display the result and the stats
print(result_df)
print(query_stats)

# Close the connection
conn.close()

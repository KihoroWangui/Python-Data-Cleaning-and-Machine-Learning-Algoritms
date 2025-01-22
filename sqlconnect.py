import oracledb

# Connect to Oracle
connection = oracledb.connect(
    user="Hannah",
    password="tuk123",
    
)

# Query the database
cursor = connection.cursor()
cursor.execute("SELECT * Hannah")
for row in cursor:
    print(row)

connection.close()
